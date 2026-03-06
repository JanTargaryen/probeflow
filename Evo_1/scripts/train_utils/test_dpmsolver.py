import torch
import math

def eval_velocity(x, t):
    # Target ODE: dx/dt = -2 * x * t
    return -2.0 * x * t

def exact_solution(x0, t):
    # Analytical solution: x(t) = x0 * exp(-t^2)
    return x0 * math.exp(-t**2)

def euler_solver(x0, steps):
    x = torch.tensor([x0], dtype=torch.float32)
    dt = 1.0 / steps
    t = 0.0
    for _ in range(steps):
        v = eval_velocity(x, t)
        x = x + v * dt
        t += dt
    return x.item()

def heun_solver(x0, steps):
    x = torch.tensor([x0], dtype=torch.float32)
    dt = 1.0 / steps
    t = 0.0
    for _ in range(steps):
        v1 = eval_velocity(x, t)
        x_mid = x + v1 * dt
        v2 = eval_velocity(x_mid, t + dt)
        x = x + 0.5 * (v1 + v2) * dt
        t += dt
    return x.item()

def dpm_multistep_solver(x0, steps):
    x = torch.tensor([x0], dtype=torch.float32)
    dt = 1.0 / steps
    t = 0.0
    v_prev = None
    for i in range(steps):
        v_current = eval_velocity(x, t)
        if i == 0:
            x = x + v_current * dt
        else:
            x = x + (1.5 * v_current - 0.5 * v_prev) * dt
        v_prev = v_current
        t += dt
    return x.item()

def run_verification():
    x0 = 1.0
    t_end = 1.0
    exact_val = exact_solution(x0, t_end)
    
    step_sizes = [10, 20, 40]
    
    print(f"Exact Solution at t={t_end}: {exact_val:.6f}\n")
    print(f"{'Steps':<8} | {'Euler Err':<12} | {'Heun Err':<12} | {'Multistep Err':<12}")
    print("-" * 55)
    
    for steps in step_sizes:
        euler_val = euler_solver(x0, steps)
        heun_val = heun_solver(x0, steps)
        multi_val = dpm_multistep_solver(x0, steps)
        
        euler_err = abs(euler_val - exact_val)
        heun_err = abs(heun_val - exact_val)
        multi_err = abs(multi_val - exact_val)
        
        print(f"{steps:<8} | {euler_err:.6f}     | {heun_err:.6f}     | {multi_err:.6f}")

if __name__ == "__main__":
    run_verification()