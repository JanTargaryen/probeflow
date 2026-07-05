"""
AdaFlow 风格方差加权负对数似然（NLL）损失函数。

论文参考: AdaFlow: Imitation Learning with Variance-Adaptive Flow-Based Policies
(Hu et al., NeurIPS 2024)

损失公式（每个样本）:
    L_i = 1/(2 · σ²) · ||v_pred - v_target||² + log(σ)

其中:
    σ² = exp(2 · log_sqrt_var)   — 模型预测的方差
    ||·||² = 所有动作维度的平方和（不是均值！）
    log_sqrt_var = log(√σ²) = 0.5 · log(σ²)

关键实现细节:
    误差在维度上是求和（SUM）而非求平均（MEAN）。
    这匹配原始 AdaFlow 代码:
        error = (target - pred).pow(2).sum(dim=(-1, -2))  # 所有维度求和

    使用 SUM 而非 MEAN 的原因:
    如果用 MEAN，最优 log_sigma = 0.5·log(每维MSE)，当 MSE < 1 时为负数，
    导致方差头总是预测低方差 → 步数卡在最小值。
    用 SUM 后，最优 log_sigma = 0.5·log(D · 每维MSE)，当 D > 1/MSE 时为正数，
    方差头可以预测合理的高方差 → 模型不确定时步数增多。

梯度分析:
    dL/d(log_σ) = 1 - ||error||² / σ²
    最优时: σ² = ||error||², loss = 0.5 + 0.5·log(||error||²)
"""
import torch


def adaflow_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    log_sqrt_var: torch.Tensor,
    action_mask: torch.Tensor = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    计算方差加权 NLL 损失（AdaFlow 论文公式）。

    Args:
        pred_velocity: (B, D) 模型预测的速度 v_θ(x, t)
        target_velocity: (B, D) 真实速度 = actions_gt - noise
        log_sqrt_var: (B,) 每样本的 log(√方差) = 0.5·log(σ²)
            方差头输出，代表模型对当前预测的不确定性
        action_mask: (B, D) 可选的二值掩码，用于标记 padding 的动作维度
        reduction: "mean" 或 "sum"，控制 batch 维度的聚合方式

    Returns:
        loss: 标量损失值
    """
    # === 步骤 1: 计算预测误差的平方 ===
    error = (pred_velocity - target_velocity).pow(2)  # (B, D)

    if action_mask is not None:
        # 如果有掩码，将 padding 位置的误差置零
        mask = action_mask.view_as(error).to(dtype=error.dtype)
        error = error * mask
        # ★★★ 关键: 所有维度求和（SUM），不是求平均（MEAN）★★★
        # 匹配原始 AdaFlow: error.sum(dim=(-1, -2))
        # 确保方差尺度与动作维度 D 成正比
        per_sample_error = error.sum(dim=1)  # (B,)
    else:
        per_sample_error = error.sum(dim=1)  # (B,)

    # === 步骤 2: 计算方差 ===
    # σ² = exp(2 · log_sqrt_var)
    var = torch.exp(2 * log_sqrt_var)  # (B,)

    # === 步骤 3: 计算 NLL 损失 ===
    # L = 1/(2·σ²) · ||error||² + log(σ)
    # 其中 log(σ) = log_sqrt_var
    per_sample_loss = 0.5 * per_sample_error / var + log_sqrt_var

    if reduction == "mean":
        return per_sample_loss.mean()
    elif reduction == "sum":
        return per_sample_loss.sum()
    else:
        return per_sample_loss
