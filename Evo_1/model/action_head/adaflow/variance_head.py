"""
AdaFlow 风格方差头（Variance Head）。

从 Flow Matching 动作头的池化特征中预测 log(√方差)。

架构: 4层 MLP (input_dim → 512 → 512 → 512 → 1)，SiLU 激活。
与原始 AdaFlow 论文的方差估计器一致:
    log σ̂(x) = MLP(h_bottleneck)

原始 AdaFlow 源码 (conditional_unet1d.py):
    self.var_est = nn.Sequential(
        nn.Linear(input_dim, 512), nn.SiLU(),
        nn.Linear(512, 512),       nn.SiLU(),
        nn.Linear(512, 512),       nn.SiLU(),
        nn.Linear(512, 1),
    )

训练阶段:
    方差头通过 NLL 损失训练，base model 冻结:
        L = 1/(2·σ²)·||v - v_θ||² + log(σ)
    这鼓励 σ² 匹配预测误差 ||v - v_θ||²。
    误差大 → 方差大 → 步长小 → 推理时 Euler 步数多。

推理阶段:
    log_sqrt_var 用于逐步自适应步长:
        σ = exp(log_sqrt_var)
        Δt = max(η/σ, 1/N_max)
    方差低（模型自信，流形线性）→ Δt 大 → 步数少
    方差高（模型不确定，流形弯曲）→ Δt 小 → 步数多

论文参考: AdaFlow: Imitation Learning with Variance-Adaptive Flow-Based Policies
(Hu et al., NeurIPS 2024)
"""
import torch
import torch.nn as nn


class AdaFlowVarianceHead(nn.Module):
    """
    4层 MLP 方差估计器（与原始 AdaFlow 架构一致）。

    输入: x_pooled — 动作头 Transformer 的池化输出特征
    输出: log_sqrt_var = log(√σ²)，其中 σ² 是预测方差

    架构对照原始 AdaFlow 的 ConditionalUnet1DwithVarianceEstimation:
        Linear(input_dim, 512) → SiLU → Linear(512, 512) → SiLU →
        Linear(512, 512) → SiLU → Linear(512, 1)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 输入层: embed_dim → 512
            nn.SiLU(),                           # 激活函数
            nn.Linear(hidden_dim, hidden_dim),  # 隐藏层 1: 512 → 512
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 隐藏层 2: 512 → 512
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),           # 输出层: 512 → 1
        )

    def forward(self, x_pooled: torch.Tensor) -> torch.Tensor:
        """
        从池化特征预测 log(√方差)。

        Args:
            x_pooled: (B, embed_dim) 动作头的池化特征。
                这是 Transformer 输出的均值池化结果，
                类似于原始 AdaFlow 中 U-Net 瓶颈层特征。

        Returns:
            log_sqrt_var: (B,) 每样本的 log(√方差)。
                σ = exp(log_sqrt_var) 是预测的标准差，
                推理时用于自适应步长: Δt = max(η/σ, 1/N_max)
        """
        log_sqrt_var = self.net(x_pooled).squeeze(-1)
        return log_sqrt_var
