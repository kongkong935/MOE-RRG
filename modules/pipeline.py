# pipeline.py
import torch, torch.nn.functional as F
import modules.utils as utils


def train(self,fc_feats,att_feats,view_ids,visit_ids,targets,prefix_kv=None,att_masks=None):
    # === 1. 准备输入特征和掩码 ===
    # 包括：给 att_feats 加位置编码、构造 decoder 输入 seq 和 mask
    att_feats, att_masks, seq, seq_mask = self._prepare_feature_forward(att_feats, att_masks, targets)

    # === 2. 编码器前向传播（提取上下文 memory）===
    encoder_output,loss_moe = self.transformer_block.encode(att_feats, att_masks, view_ids, visit_ids,train=True)

    # === 3. 解码器前向传播（自回归生成 token logits）===
    out = self.transformer_block.decode(memory=encoder_output,src_mask=att_masks,tgt=seq,tgt_mask=seq_mask,
                                        prefix_kv=prefix_kv,memory_matrix=self.memory_matrix)
    # === 4. 映射词表空间并取 log-prob ===
    word_out = F.log_softmax(self.logit(out), dim=-1)  # (B, T-1, V)

    return word_out,out,loss_moe


# ---------- 推理 ----------
def sample(self,fc_feats,att_feats,view_ids,visit_ids,prefix_kv=None,att_masks=None,update_opts={}):
    """
    核心流程：准备特征 → 第一步取 logprobs → Beam Search 主循环 → 汇总输出
    ----------------------------------------------------------------------
    返回:
        seq         (B, max_len)          : 生成的 token 序列
        seqLogprobs (B, max_len, vocab+1) : 对应 log 概率（可选做 rerank）
    """

    # ------------------------------------------------------------------ #
    # 0⃣️  解析推理参数（束宽 / 采样数等）                                 #
    # ------------------------------------------------------------------ #
    opt         = self.args.__dict__
    opt.update(update_opts)                     # 允许外部覆盖
    beam_size   = opt.get('beam_size', 3)      # 默认束宽
    group_size  = opt.get('group_size', 1)      # 分组采样（一般 =1）
    sample_n    = opt.get('sample_n', 10)       # Beam 内要返回几个样本

    assert sample_n == 1 or sample_n == beam_size // group_size, \
        "当使用 beam search 时 sample_n 必须为 1 或 beam_size/group_size"
    assert beam_size <= self.vocab_size + 1, \
        "beam_size 过大，> vocab_size + 1 会造成 corner case"

    batch_size = self.args.batch_size
    # ------------------------------------------------------------------ #
    # 1⃣️  准备特征 + 编码器前向（含 MoE 路由）                           #
    # ------------------------------------------------------------------ #
    #  1.1 🎛 位置编码、mask 等常规预处理
    att_feats, att_masks, seq, seq_mask = self._prepare_feature_forward(att_feats, att_masks)

    #  1.2 🚦 进入带 MoE-Router 的 Encoder
    memory = self.transformer_block.encode(att_feats,att_masks,view_ids, visit_ids,train=False)   # 推理模式
    #  1.3 取出 BeamSearch 所需最简占位张量
    p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = fc_feats[..., :1], att_feats[..., :1], memory, att_masks


    # ------------------------------------------------------------------ #
    # 2⃣️  初始化 Beam Search 容器                                        #
    # ------------------------------------------------------------------ #
    seq         = fc_feats.new_full((batch_size * sample_n, self.max_seq_length),self.pad_idx, dtype=torch.long)
    seqLogprobs = fc_feats.new_zeros(batch_size * sample_n,self.max_seq_length,self.vocab_size + 1)
    self.done_beams = [[] for _ in range(batch_size)]     # 存储每张图的 beam
    state = self.init_hidden(batch_size)                  # 初始解码器隐状态

    # ------------------------------------------------------------------ #
    # 3⃣️  第 0 步：输入 <bos>，计算首个 logprobs，内部调用core然后调用decoder            #
    # ------------------------------------------------------------------ #
    it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)

    # print("🆕第一次执行get_logprobs_state")
    logprobs, state = self.get_logprobs_state(it,p_fc_feats, p_att_feats,pp_att_feats, p_att_masks, state,prefix_kv)

    # 复制特征到每个 beam（显存充足可保留；显存紧张可改懒复制）
    if prefix_kv is not None:
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_prefix_k, p_prefix_v = utils.repeat_tensors(beam_size,
                    [p_fc_feats,p_att_feats,pp_att_feats,p_att_masks,prefix_kv[0],prefix_kv[1]])
        prefix_kv = torch.stack([p_prefix_k, p_prefix_v])
    else:
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
                    [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks])



    # ------------------------------------------------------------------ #
    # 4⃣️  调用统一的 beam_search() 主循环                                #
    # ------------------------------------------------------------------ #
    # print("开始beamsearch")
    self.done_beams = self.beam_search(state, logprobs, prefix_kv, p_fc_feats, p_att_feats,pp_att_feats, p_att_masks,opt=opt)


    # ------------------------------------------------------------------ #
    # 5⃣️  汇总 beam 结果 → seq / seqLogprobs                            #
    # ------------------------------------------------------------------ #
    for k in range(batch_size):
        if sample_n == beam_size:                          # 每个 beam 都要
            for _n in range(sample_n):
                seq_len  = self.done_beams[k][_n]['seq'].shape[0]
                seq[k * sample_n + _n, :seq_len]          = self.done_beams[k][_n]['seq']
                seqLogprobs[k * sample_n + _n, :seq_len]  = self.done_beams[k][_n]['logps']
        else:                                              # 只取 beam-0
            seq_len  = self.done_beams[k][0]['seq'].shape[0]
            seq[k, :seq_len]         = self.done_beams[k][0]['seq']
            seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']


    # ------------------------------------------------------------------ #
    # 6⃣️  返回生成序列及其 log 概率                                      #
    # ------------------------------------------------------------------ #
    return seq, seqLogprobs


    #
    # # ===== 解码参数设置 =====
    # beam_size = opt.get('beam_size', 1)# beam search 的束宽（
    # group_size = opt.get('group_size', 1)# 分组采样（diverse beam search）的分组数量。用于提升生成文本的多样性。
    #
    # # ===== beam/diverse 搜索直接转交原函数 =====
    # if beam_size > 1:
    #     return self._sample_beam(fc_feats, att_feats,view_ids,visit_ids, att_masks, opt)
    # if group_size > 1:
    #     return self._diverse_sample(fc_feats, att_feats, att_masks, opt)
