from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moe_router import MoEGateRouter
from .att_model import pack_wrapper, AttModel


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    selected_scores, idx = scores.topk(topk)
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    selected_value = torch.gather(dummy_value, 3, dummy_idx)
    p_attn = F.softmax(selected_scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, cmn):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.cmn = cmn

    # def forward(self, src, tgt, src_mask, tgt_mask, memory_matrix):
    #     return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, memory_matrix=memory_matrix)

    # def encode(self, src, src_mask):
    #     return self.encoder(self.src_embed(src), src_mask)

    # def decode(self, memory, src_mask, tgt, tgt_mask,  past=None, memory_matrix=None):
    #     embeddings = self.tgt_embed(tgt)
    #
    #     # Memory querying and responding for textual features
    #     dummy_memory_matrix = memory_matrix.unsqueeze(0).expand(embeddings.size(0), memory_matrix.size(0), memory_matrix.size(1))
    #     responses = self.cmn(embeddings, dummy_memory_matrix, dummy_memory_matrix)
    #     embeddings = embeddings + responses
    #     # Memory querying and responding for textual features
    #
    #     return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past)

    def encode(self, src, src_mask, view_ids, visit_ids, train=True):
        return self.encoder(self.src_embed(src), src_mask,view_ids, visit_ids, train=train)

    # out = self.transformer_block.decode(memory=encoder_output,src_mask=att_masks,tgt=seq,tgt_mask=seq_mask,
    #                                     prefix_kv=prefix_kv,memory_matrix=self.memory_matrix)

    # out, past = self.transformer_block.decode(memory=memory, src_mask=mask, tgt=tgt, tgt_mask=tgt_mask,
    #                                     prefix_kv=prefix_kv,past=past, memory_matrix=self.memory_matrix)
    def decode(self, memory, src_mask, tgt, tgt_mask,prefix_kv, past=None, memory_matrix=None):
        embeddings = self.tgt_embed(tgt)
        # Memory querying and responding for textual features
        dummy_memory_matrix = memory_matrix.unsqueeze(0).expand(embeddings.size(0), memory_matrix.size(0), memory_matrix.size(1))
        responses = self.cmn(embeddings, dummy_memory_matrix, dummy_memory_matrix)
        embeddings = embeddings + responses
        # Memory querying and responding for textual features

        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past,prefix_kv=prefix_kv)


# class Encoder(nn.Module):
#     def __init__(self, layer, N):
#         super(Encoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.size)
#
#     def forward(self, x, mask):
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)


class MoELayer(nn.Module):
    """
    通用 MoE 模块：基于 view_id + visit_bucket 路由，选择专家进行前馈。
    """
    def __init__(self, d_model, n_experts, router: nn.Module):
        super().__init__()
        self.router = router  # MoEGateRouter 实例
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(n_experts)
        ])
        self.balance_loss= 0.0


    def forward(self, x, view_ids, visit_ids, train=True):
        """
        参数：
            x:         (B, L, D) 输入特征
            view_ids:  (B,) 视角 ID
            visit_ids: (B,) 就诊顺序 ID
        返回：
            out:       (B, L, D) 各专家处理后的输出，按样本合并
        """
        logits = self.router(view_ids, visit_ids)       # (B, n_experts)
        top1 = logits.argmax(dim=-1)                    # (B,) 每个样本选一个专家

        # ❗只有在训练时才计算 balance_loss
        if train:
            probs = logits.softmax(dim=-1)
            load = probs.mean(0)                        # 每个专家平均被选概率
            self.balance_loss = -(load * load.log()).sum()
        else:
            self.balance_loss = 0.0

        out = torch.zeros_like(x)                       # 初始化输出

        # 遍历每个专家，选择它负责的样本执行前馈
        for eid in range(len(self.experts)):
            selected_mask = (top1 == eid)               # shape: (B,)
            if selected_mask.sum() == 0:
                continue

            x_selected = x[selected_mask]               # (N, L, D)
            y_selected = self.experts[eid](x_selected)  # (N, L, D)

            out[selected_mask] = y_selected             # 直接用索引回写

        return out
class Encoder(nn.Module):
    def __init__(self, layer, N,
                 moe_layer_cls=None,           # =MoELayer
                 router=None,                  # MoEGateRouter 实例
                 n_experts=4,
                 moe_indices=(1, 3, 5)):       # 在哪些层前插 MoE
        super().__init__()
        self.layers = clones(layer, N)
        self.norm   = LayerNorm(layer.size)

        # === 新增 MoE 容器 ===
        self.moe_indices = set(moe_indices)
        if moe_layer_cls is not None:
            self.moe_layers = nn.ModuleDict({
                str(idx): moe_layer_cls(layer.size, n_experts, router)
                for idx in self.moe_indices
            })

        else:
            self.moe_layers = None

    def forward(self, x, mask, view_ids=None, visit_ids=None, train=True):
        batch_moe_loss = torch.tensor(0., device=x.device)
        for idx, layer in enumerate(self.layers):
            # --- 若命中 MoE 插槽，先走 MoE ---
            if self.moe_layers and idx in self.moe_indices:
                moe_layer = self.moe_layers[str(idx)]
                x = moe_layer(x, view_ids, visit_ids, train=train)
                # 把 balance_loss 暴露出去（可累加到总 loss）
                if self.training:
                    batch_moe_loss = batch_moe_loss + moe_layer.balance_loss

            # --- 原始 EncoderLayer 前向 ---
            x = layer(x, mask)
        if train:
            return self.norm(x), batch_moe_loss  # 训练 → 返回 loss
        else:
            return self.norm(x)  # 推理 → 只返回 memory




class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

#全连接层
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# class Decoder(nn.Module):
#     def __init__(self, layer, N):
#         super(Decoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.size)
#
#     def forward(self, x, memory, src_mask, tgt_mask, past=None):
#         if past is not None:
#             present = [[], []]
#             x = x[:, -1:]
#             tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
#             past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
#         else:
#             past = [None] * len(self.layers)
#         for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
#             x = layer(x, memory, src_mask, tgt_mask,
#                       layer_past)
#             if layer_past is not None:
#                 present[0].append(x[1][0])
#                 present[1].append(x[1][1])
#                 x = x[0]
#         if past[0] is None:
#             return self.norm(x)
#         else:
#             return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)]

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        # 所有层都注入同一份 prefix_kv,也可以单独设计那些层注入
        self.inject_layers = set(range(N))

    def forward(self,x,memory,src_mask,tgt_mask,prefix_kv,past=None):    # 来自缓存的 past key/value
        """
        prefix_kv: optional tensor [2, B, M, D], 0->key cache, 1->value cache
        past:       list of length N, 每层的 (key, value) tuple
        """
        # —— 1. 处理 past 缓存 —— #也就sample模式
        if past is not None:
            present = [[], []]
            x = x[:, -1:]  # keep last step
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            # 还原每层的 key/value
            past = list(zip(past[0].split(2, dim=0),
                            past[1].split(2, dim=0)))
        else:#也就是train模式
            present = None
            past = [None] * len(self.layers)

        # —— 2. 逐层解码 + 注入 prefix_kv —— #
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            if  i in self.inject_layers:
                x_out = layer(x, memory, src_mask, tgt_mask,prefix_kv,layer_past)
            else:
                x_out = layer(x, memory, src_mask, tgt_mask, layer_past)

            # 如果 layer 返回 (output, (key, value))
            if isinstance(x_out, tuple):
                x, present_kv = x_out
                if present is not None:
                    present[0].append(present_kv[0])  # key
                    present[1].append(present_kv[1])  # value
            else:
                x = self.norm(x_out)

        if present is None:# train模式的返回
            return x
        else:# sample模式的返回,把每层缓存 concat 回去，形成新的 past
            return x, [torch.cat(present[0], dim=0),torch.cat(present[1], dim=0)]


# class DecoderLayer(nn.Module):
#     def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
#         super(DecoderLayer, self).__init__()
#         self.size = size
#         self.self_attn = self_attn
#         self.src_attn = src_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 3)
#
#     def forward(self, x, memory, src_mask, tgt_mask,prefix_kv, layer_past=None):
#         m = memory
#         if layer_past is None:
#             x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
#             x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
#             return self.sublayer[2](x, self.feed_forward)
#         else:
#             present = [None, None]
#             x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
#             x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
#             return self.sublayer[2](x, self.feed_forward), present

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, prefix_kv=None, layer_past=None):
        """
        单层解码器，支持在 self-attn 阶段注入 prefix_kv（视觉前缀 Key/Value）。

        Args:
            x:           Tensor [B, T, D], 当前解码器输入，训练时 T=seq_len，推理时 T=1
            memory:      Tensor [B, S, D], 编码器输出
            src_mask:    Tensor, 编码器掩码
            tgt_mask:    Tensor, 解码器自注意力掩码
            layer_past:  2个一个是存储的 Mask-MHA的 KV,一个存储的 Cross-MHA的 KV
            prefix_kv:   Optional Tensor [2, B, M, D], 0=key prefix, 1=value prefix
        Returns:
            如果 layer_past 为 None:
                Tensor [B, T, D]
            否则:
                (Tensor [B, T, D], present) 其中 present=[Mask-MHA的 KV, Cross-MHA的 KV]
        """
        B = tgt_mask.shape[0]
        T = tgt_mask.shape[1]
        # 训练模式,在外面直接注入到 key和 value,还是只是在 MHA 注入
        if layer_past is None:
            if prefix_kv is not None:#这一层注入了 prefix_kv
                M = prefix_kv.shape[2]
                prefix_mask = torch.ones((B, T, M), dtype=tgt_mask.dtype, device=tgt_mask.device)
                tgt_mask = torch.cat([prefix_mask, tgt_mask], dim=-1)  # [B, T, M+T]
                #print(tgt_mask)
                # 拼接 prefix 和输入 x 作为 key/value
                key = torch.cat([prefix_kv[0], x], dim=1)  # [B, M+T, D]
                value = torch.cat([prefix_kv[1], x], dim=1)
                x = self.sublayer[0](x,lambda x: self.self_attn(x, key, value, mask=tgt_mask))# 执行 self-attn：Query=x, Key=key, Value=value
            else:# 无 prefix
                x = self.sublayer[0](x,lambda x: self.self_attn(x, x, x, mask=tgt_mask)[0])
            # 2. Cross-Attn:编码器-解码器交叉注意力
            x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, mask=src_mask))
            return self.sublayer[2](x, self.feed_forward)

        # 推理模式:使用 layer_past加速,在内部根据逻辑注入到 key和 value
        else:
            present = [None, None]
            if prefix_kv is not None:
                M = prefix_kv.shape[2]
                prefix_mask = torch.ones((B, T, M), dtype=tgt_mask.dtype, device=tgt_mask.device)
                tgt_mask = torch.cat([prefix_mask, tgt_mask], dim=-1)  # [B, T, M+T]
                # print(f"调用了MHA:")
                x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x,mask=tgt_mask,prefix_kv=prefix_kv,layer_past=layer_past[0]))

            else:
                x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x,mask=tgt_mask,layer_past=layer_past[0]))
            # 2. Cross-Attn:编码器-解码器交叉注意力
            # print(f"调用了CA:")
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x,memory,memory,mask=src_mask,layer_past=layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present


#         if layer_past is None:
#             x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
#             x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
#             return self.sublayer[2](x, self.feed_forward)
#         else:
#             present = [None, None]
#             x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
#             x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
#             return self.sublayer[2](x, self.feed_forward), present
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value , mask=None, prefix_kv=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        #beamsearch前运行一次拿到缓存 CA就一直走这里了,不需要考虑了
        if layer_past is not None and  (layer_past.shape[2] == key.shape[1] > 1):
            # print("🔁 走了缓存")
            # print(f"layer_past.shape[2] = {layer_past.shape[2]}, key.shape[1]  = {key.shape[1]}")
            # print("------------------------------------------------------------------------------")
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
        #encoder使用,beamsearch前一次使用,beamsearch中 MHA使用
            # print("🆕 没有拿到缓存，执行 QKV 线性变换")
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]
        # 走 MHA的时候,就会走这里,因为 key.shape[1]一直都是1,layer_past.shape[2]是累计长度(为 1的时候单独判断 )
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
        # if layer_past is not None:
        # print(f"条件不足，缓存+当前一行：layer_past.shape[2] = {layer_past.shape[2]}, key.shape[1]  = {key.shape[1]}")
        # print("")
            past_key, past_value = layer_past[0], layer_past[1]
            # 拼接记得次序要和 mask对齐
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])
        if prefix_kv is not None:#只在 MHA中做信息注入
            # print("注入了辅助信息.")
            prefix_key,prefix_value = prefix_kv[0],prefix_kv[1]
            #拼接记得次序要和 mask对齐
            key = torch.cat((prefix_key, key), dim=1)
            value = torch.cat((prefix_value, value), dim=1)


        query, key, value = \
            [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.h * self.d_k)

        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)




class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32):
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)
#FFN层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)





class BaseCMN(AttModel):

    def make_model(self, tgt_vocab, cmn):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        router = MoEGateRouter(n_experts=4, view_vocab_size=4, visit_vocab_size=5)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers,
                    router=router, n_experts=4, moe_indices=(2, 5)),  # 驱动第3、6层加 MoE，可自行调整),
            # Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout),self.num_layers,
            #     moe_layer_cls=MoELayer,router=router,n_experts=4,moe_indices=(2, 5)),  # 驱动第3、6层加 MoE，可自行调整),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            nn.Sequential(c(position)),
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            cmn)
        #print(model)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer ,):
        super(BaseCMN, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.topk = args.topk
        tgt_vocab = self.vocab_size + 1
        self.cmn = MultiThreadMemory(args.num_heads, args.d_model, topk=args.topk)
        self.transformer_block = self.make_model(tgt_vocab, self.cmn)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

        self.memory_matrix = nn.Parameter(torch.FloatTensor(args.cmm_size, args.cmm_dim))
        nn.init.normal_(self.memory_matrix, 0, 1 / args.cmm_dim)

    def init_hidden(self, bsz):
        return []

# # sample 阶段准备数据的函数
#     def _prepare_feature(self, fc_feats, att_feats,view_ids,visit_ids, att_masks):
#         att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
#         memory = self.transformer_block.encode(att_feats, att_masks,view_ids,visit_ids,train=False)
#
#         return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

#作用：进入encoder前准备数据
    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)

        # Memory querying and responding for visual features
        dummy_memory_matrix = self.memory_matrix.unsqueeze(0).expand(att_feats.size(0), self.memory_matrix.size(0), self.memory_matrix.size(1))
        responses = self.cmn(att_feats, dummy_memory_matrix, dummy_memory_matrix)
        att_feats = att_feats + responses
        # Memory querying and responding for visual features

        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, att_masks, seq, seq_mask

    # #训练模式激活
    # def _forward(self, fc_feats, att_feats, seq, att_masks=None):
    #     att_feats, att_masks, seq, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
    #     out = self.transformer_block(att_feats, seq, att_masks, seq_mask, memory_matrix=self.memory_matrix)
    #     outputs = F.log_softmax(self.logit(out), dim=-1)

        return outputs

    def _save_attns(self, start=False):
        if start:
            self.attention_weights = []
        self.attention_weights.append([layer.src_attn.attn.cpu().numpy() for layer in self.transformer_block.decoder.layers])

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask, prefix_kv):
        if len(state) == 0:
            ys = it.unsqueeze(1)
            past = [fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model),
                    fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model)]
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
            past = state[1:]
        tgt=ys
        tgt_mask=subsequent_mask(ys.size(1)).to(memory.device)
        out, past = self.transformer_block.decode(memory=memory,src_mask=mask, tgt=tgt, tgt_mask=tgt_mask,
                                                  prefix_kv=prefix_kv,past=past,memory_matrix=self.memory_matrix)

        if not self.training:
            self._save_attns(start=len(state) == 0)
        return out[:, -1], [ys.unsqueeze(0)] + past
