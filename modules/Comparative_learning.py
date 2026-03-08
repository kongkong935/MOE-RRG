# === imploss_module.py ===============================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchEncoding
from modules.Attentivedsentencepool import AttentiveSentencePool, ProjectionHead


class ImpressionContrastiveLoss(nn.Module):
    """
    计算生成文本 (gen) 与 Impression (imp) 的 InfoNCE 对比损失
    --------------------------------------------------------------
    - 正对：同 index 的 (gen_i, imp_i)
    - 负对：batch 其余 (gen_i, imp_j≠i)
    """
    def __init__(self,bert_tokenizer,bert,tokenizer,tau):
        super().__init__()
        self.bert = bert
        self.bert_tokenizer =bert_tokenizer
        self.tokenizer = tokenizer
        self.tau = tau
        self.pooler= AttentiveSentencePool(d_model=512)
        self.proj_gen = ProjectionHead(512, 256)  # 用于 gen_vec
        self.proj_bert = ProjectionHead(768, 256)  # 用于 imp_vec

    # -------- 前向：计算 InfoNCE --------
    def forward(self, gen_logits, imp_ids, imp_mask,device):
        """
        gen_logits: (B, T, V) — logits from decoder
        imp_ids:    dict with 'input_ids': (B, T_imp) from tokenizer
        imp_mask:   (B, T_imp) — 0 for [NIMP], 1 for valid
        """
        # === Step 1: 过滤掉空 Impression 的样本 ===
        if imp_mask.sum() == 0:
            return torch.tensor(0.0, device=gen_logits.device)

        # === Step 2: 生成 token ids from logits ===
        valid_idx = (imp_mask != 0).nonzero(as_tuple=True)[0]

        imp_ids = filter_batch_encoding(imp_ids, valid_idx)# 提取imp有效样本
        imp_vec     = self.bert(**imp_ids).last_hidden_state[:, 0]


        gen_ids = gen_logits[valid_idx]  # 提取imp有效样本
        gen_vec = self.pooler(gen_ids)  # 池化拿到信息.


        # ① 归一化
        i_vec = F.normalize(self.proj_bert(imp_vec), dim=-1)
        r_vec = F.normalize(self.proj_gen(gen_vec), dim=-1)

        # ② 相似度矩阵  [B, B]
        logits = i_vec @ r_vec.T / self.tau

        # ③ 行方向 InfoNCE
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2r = F.cross_entropy(logits, labels)
        # ④ 列方向再算一遍，求平均
        loss_r2i = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_i2r + loss_r2i)



# =====================================================================
# === compute_loss_imp.py（与 compute_loss 并列的简单包装） ===========
def compute_loss_imp(output, imp_ids, imp_mask,c_loss_imp,device):
    """
    output  : (B, T, V) logits  *或*  (B, T) already token ids
    imp_ids : (B, T_imp)  impression token ids
    imp_mask: (B, T_imp)  1=valid, 0=pad
    """
    return c_loss_imp(output, imp_ids, imp_mask,device)
# =====================================================================
#
# def contrastive_loss(i_vec, r_vec, tau=0.07):
#     """
#     i_vec, r_vec: [B, D]  — 已经过 BERT 得到的句向量
#     返回标量 loss
#     """
#     # ① 归一化
#     i_vec = F.normalize(i_vec, dim=-1)
#     r_vec = F.normalize(r_vec, dim=-1)
#
#         # ② 相似度矩阵  [B, B]
#     logits = i_vec @ r_vec.T / tau
#
#         # ③ 行方向 InfoNCE
#     labels = torch.arange(logits.size(0), device=logits.device)
#     loss_i2r = F.cross_entropy(logits, labels)
#         # ④ 列方向再算一遍，求平均
#     loss_r2i = F.cross_entropy(logits.T, labels)
#     return 0.5 * (loss_i2r + loss_r2i)


def filter_batch_encoding(batch_encoding, valid_idx):
    """
    保留 BatchEncoding 中每个 tensor 字段的有效样本（按 batch 第一维过滤）
    返回仍为 BatchEncoding 类型，兼容 .to(), .keys(), 可直接传入模型
    """
    filtered = {
        k: v[valid_idx] for k, v in batch_encoding.items() if isinstance(v, torch.Tensor)
    }
    return BatchEncoding(filtered)

