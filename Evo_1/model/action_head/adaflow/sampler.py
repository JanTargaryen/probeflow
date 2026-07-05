"""
AdaFlow 风格自适应采样器（推理阶段）。

与原始 AdaFlow 论文的逐步自适应策略一致。
不同于"只算一次方差，用固定步长"的简化方案，此实现
在每一步都重新计算方差并动态调整步长。

论文: AdaFlow: Imitation Learning with Variance-Adaptive Flow-Based Policies
(Hu et al., NeurIPS 2024)

每步更新（论文 Algorithm 1）:
    1. 计算速度:          v = v_θ(x_t, t)
    2. 预测方差:          σ = exp(log_sqrt_var(x_t, t))
    3. 自适应步长:        Δt = max(η / σ, 1/N_max)
    4. Euler 更新:        x_{t+Δt} = x_t + v · Δt
    5. 时间推进:          t ← t + Δt
    6. 提前停止:          t ≥ 1 时停止

与固定步长 Euler 的关键区别:
    - 每步重新计算方差（不只是 t=0 时算一次）
    - 步长根据当前不确定性自适应调整（不是固定 1/N）
    - 样本到达 t=1 后独立停止
"""
import torch


@torch.no_grad()
def adaflow_sample(
    model,
    action: torch.Tensor,
    context_tokens: torch.Tensor,
    embodiment_id: torch.LongTensor,
    action_mask_seq: torch.Tensor = None,
    per_action_dim: int = 7,
    eta: float = 0.5,
    min_steps: int = 2,
    max_steps: int = 50,
    verbose: bool = False,
) -> tuple:
    """
    逐步自适应采样（AdaFlow Algorithm 1）。

    每一步:
        1. σ = exp(variance_head(pooled_features))    — 预测不确定性
        2. Δt = max(η / σ, 1/max_steps)               — 自适应步长
        3. x ← x + v_θ(x, t) · Δt                     — Euler 更新
        4. t ← t + Δt                                  — 时间推进
        5. 如果 t ≥ 1: 停止                            — 提前终止

    方差 σ = exp(log_sqrt_var) 代表模型对速度预测的不确定性。
    不确定性高 → 步长小 → 步数多（更保守的积分）。

    Args:
        model: 带有 variance_head 的 FlowmatchingActionHead。需提供:
            - model.eval_velocity(action, t, ...): 计算 (x, t) 处的速度
            - model.variance_head(x_pooled): 预测 log_sqrt_var
            - model._cached_x_pooled: eval_velocity 设置的池化特征
        action: (B, D) 初始噪声（标准高斯分布）
        context_tokens: (B, N, D) 视觉-语言上下文特征
        embodiment_id: (B,) 本体类别 ID
        action_mask_seq: (B, H, P) 可选动作掩码（None = 全部有效）
        per_action_dim: 每个时间步的动作维度
        eta: 自适应步长缩放因子
            大 η → 大步长 → 总步数少（更激进）
            小 η → 小步长 → 总步数多（更保守）
            典型范围: 0.5-2.0
        min_steps: 最小步数（步长上限: Δt ≤ 1/min_steps）
        max_steps: 最大步数（步长下限: Δt ≥ 1/max_steps，安全限制）
        verbose: 打印每步调试信息

    Returns:
        action: (B, D) 在 t=1 处生成的动作
        metadata: dict 包含:
            - "steps": 实际使用的总积分步数
            - "log_sqrt_var": 所有步的 log_sqrt_var 平均值
    """
    B = action.shape[0]
    current_t = torch.zeros(B, device=action.device, dtype=action.dtype)
    step_counts = torch.zeros(B, device=action.device, dtype=torch.long)
    log_sqrt_vars = [] # 记录每步的方差

    # 如果未提供 action_mask，创建全 1 掩码
    if action_mask_seq is None:
        action_mask_seq = torch.ones(
            B, action.shape[-1] // per_action_dim, per_action_dim,
            device=action.device, dtype=action.dtype
        )

    min_dt = 1.0 / max_steps
    max_dt = 1.0 / max(min_steps, 1)

    for i in range(max_steps):
        active_mask = current_t < 1.0 - 1e-6
        if not active_mask.any():
            break

        # === [第 1 步] 计算当前时间 t 的速度 ===
        # 这也会设置 model._cached_x_pooled
        v = model.eval_velocity(
            action, current_t, context_tokens, embodiment_id,
            action_mask_seq, per_action_dim
        )

        # === [第 2 步] 从当前池化特征预测方差 ===
        # σ = exp(log_sqrt_var)  其中 log_sqrt_var = log(√σ²)
        log_sqrt_var = None
        if hasattr(model, 'variance_head') and model.variance_head is not None:
            x_pooled = getattr(model, '_cached_x_pooled', None)
            if x_pooled is not None:
                log_sqrt_var = model.variance_head(x_pooled)  # (B,)
                log_sqrt_vars.append(log_sqrt_var.mean().item())

        # === [第 3 步] 自适应步长（AdaFlow 论文公式）===
        # Δt = max(η / σ, 1/N_max)
        # σ 大（不确定）→ Δt 小 → 更保守
        # σ 小（自信）  → Δt 大 → 积分更快
        if log_sqrt_var is not None:
            sqrt_var = log_sqrt_var.exp()  # (B,) 标准差
            dt = torch.clamp(eta / sqrt_var, min=min_dt, max=max_dt)
        else:
            dt = torch.full_like(current_t, min_dt)

        # 确保不超过 t=1
        dt = torch.minimum(dt, 1.0 - current_t)
        dt = torch.where(active_mask, dt, torch.zeros_like(dt))

        # === [第 4 步] Euler 更新: x_{t+Δt} = x_t + v · Δt ===
        action = action + v * dt.unsqueeze(-1)
        current_t = current_t + dt
        step_counts = step_counts + active_mask.long()

        if verbose:
            ls = log_sqrt_var.mean().item() if log_sqrt_var is not None else 0.0
            print(
                f"  [AdaFlow t={current_t.mean().item():.3f}] "
                f"log_sqrt_var={ls:.4f}, dt={dt[active_mask].mean().item():.4f}"
            )

    # 计算所有步的平均 log_sqrt_var 用于返回
    avg_log_sqrt_var = sum(log_sqrt_vars) / len(log_sqrt_vars) if log_sqrt_vars else 0.0
    steps_value = int(step_counts[0].item()) if B == 1 else step_counts.tolist()
    metadata = {
        "steps": steps_value,
        "log_sqrt_var": avg_log_sqrt_var,
    }

    if verbose:
        print(f"  [AdaFlow] 总步数: {steps_value}, 平均 log_sqrt_var: {avg_log_sqrt_var:.4f}")

    return action, metadata
