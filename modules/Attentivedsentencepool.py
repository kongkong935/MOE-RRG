import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveSentencePool(nn.Module):
    """
    学习式 Attention-Pool
    输入 : x   [B, T, D]  —— decoder hidden
           mask[可选] [B, T]  — 1=有效token, 0=pad
    输出 : vec [B, D]     —— 句向量
    """
    def __init__(self, d_model, hidden_ratio=0.5, dropout=0.1):
        super().__init__()
        hidden_dim = int(d_model * hidden_ratio)
        self.att_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)          # 打分标量
        )

    def forward(self, x):
        device = self.att_mlp[0].weight.device  # 取线性层所处设备
        x = x.to(device)
        # x: [B,T,D]
        # 1) 计算注意力分数
        score = self.att_mlp(x).squeeze(-1)    # [B,T]

        # 2) softmax → 权重 α
        alpha = F.softmax(score, dim=-1)       # [B,T]

        # 3) 加权求和
        vec = torch.bmm(alpha.unsqueeze(1), x) # [B,1,D]
        return vec.squeeze(1)                  # [B,D]



class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=256, hidden_ratio=0.5, p=0.1):
        super().__init__()
        h = int(in_dim * hidden_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h, out_dim)
        )
    def forward(self, x):
        device = self.mlp[0].weight.device  # 取线性层所处设备
        x = x.to(device)
        return self.mlp(x)
