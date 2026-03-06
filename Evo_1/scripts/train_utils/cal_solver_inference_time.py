import torch
import math

# ==========================================================
# Dummy Network to strictly isolate solver overhead (No heavy NN)
# ==========================================================
def dummy_eval_velocity(x, t):
    # Simulate a lightweight deterministic vector field
    return -0.5 * x * t

# ==========================================================
# Solvers Implementation
# ==========================================================
def probeflow_solver(action):
    # Simulate ProbeFlow logic
    dt_probe = 0.5
    t = 0.0
    
    # Probe
    v_start = dummy_eval_velocity(action, t)
    x_mid = action + v_start * dt_probe
    v_mid = dummy_eval_velocity(x_mid, t + dt_probe)
    
    # Cosine Similarity Overhead (The main overhead of our method)
    flat_v_start = v_start.view(action.size(0), -1)
    flat_v_mid = v_mid.view(action.size(0), -1)
    cos_sim = torch.nn.functional.cosine_similarity(flat_v_start, flat_v_mid, dim=1)
    sim_score = cos_sim.min().item()
    
    curvature = 1.0 - sim_score
    epsilon = 0.008 
    raw_steps = 2 + 2 * math.floor(curvature / epsilon)
    target_steps = int(min(max(raw_steps, 2), 20))
    
    # Integration
    dt = 1.0 / target_steps
    action = action + v_start * dt
    t += dt
    
    for _ in range(target_steps - 1):
        v = dummy_eval_velocity(action, t)
        action = action + v * dt
        t += dt
        
    return action

def dpm_multistep_solver(action, steps=10):
    # DPM-Solver++ 2M equivalent (Adams-Bashforth 2)
    dt = 1.0 / steps
    t = 0.0
    v_prev = None
    
    for i in range(steps):
        v_current = dummy_eval_velocity(action, t)
        if i == 0:
            action = action + v_current * dt
        else:
            action = action + (1.5 * v_current - 0.5 * v_prev) * dt
        v_prev = v_current
        t += dt
        
    return action

def rk45_solver(action):
    # RK45 Logic with coefficient overhead and error thresholding
    rtol, atol = 1e-3, 1e-5
    t, dt = 0.0, 0.1
    
    # Massive coefficient loading
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
    e1, e3, e4, e5, e6, e7 = 71/57600, -71/16695, 71/1920, -17253/339200, 22/525, -1/40

    k1 = dummy_eval_velocity(action, t)
    while t < 1.0 - 1e-5:
        if t + dt > 1.0:
            dt = 1.0 - t
        
        # Heavy memory allocation and vector math
        k2 = dummy_eval_velocity(action + dt * (a21 * k1), t + 0.2 * dt)
        k3 = dummy_eval_velocity(action + dt * (a31 * k1 + a32 * k2), t + 0.3 * dt)
        k4 = dummy_eval_velocity(action + dt * (a41 * k1 + a42 * k2 + a43 * k3), t + 0.8 * dt)
        k5 = dummy_eval_velocity(action + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), t + (8/9) * dt)
        k6 = dummy_eval_velocity(action + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), t + dt)
        
        action_next = action + dt * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        k7 = dummy_eval_velocity(action_next, t + dt)
        
        # Error thresholding logic (CPU-GPU sync bottleneck)
        err_vec = dt * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7)
        scale = atol + rtol * torch.max(torch.abs(action), torch.abs(action_next))
        err = torch.max(torch.abs(err_vec) / scale).item()
        
        dt_next = dt * max(0.2, min(5.0, 0.9 * (err + 1e-8)**(-0.2)))
        
        if err <= 1.0:
            action = action_next
            t += dt
            k1 = k7  
        
        dt = dt_next
        
    return action

# ==========================================================
# Benchmark Framework
# ==========================================================
def measure_time(solver_fn, action_tensor, iterations=1000):
    # GPU Warmup
    for _ in range(50):
        solver_fn(action_tensor.clone())
    
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        solver_fn(action_tensor.clone())
    end_event.record()
    
    torch.cuda.synchronize()
    # Return average latency in milliseconds (ms)
    return start_event.elapsed_time(end_event) / iterations

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Simulate realistic robotic control dimensions: Batch=1, Horizon=50, ActionDim=14 -> Flat 700
    B, D = 1, 700 
    
    print(f"Benchmarking Pure Solver Overhead on {device} (B={B}, Dim={D})")
    print("-" * 65)
    
    action_init = torch.randn(B, D, device=device)
    
    # 1. Measure ProbeFlow
    probe_time = measure_time(lambda x: probeflow_solver(x), action_init)
    
    # 2. Measure DPM-Solver++ (Multistep, NFE=10)
    dpm_time = measure_time(lambda x: dpm_multistep_solver(x, steps=10), action_init)
    
    # 3. Measure RK45 (Adaptive)
    rk45_time = measure_time(lambda x: rk45_solver(x), action_init)
    
    print(f"{'Solver Method':<25} | {'Setting':<15} | {'Overhead (ms)':<15}")
    print("-" * 65)
    print(f"{'ProbeFlow (Ours)':<25} | {'Adaptive':<15} | {probe_time:.4f} ms")
    print(f"{'DPM-Solver++ 2M':<25} | {'Fixed (N=10)':<15} | {dpm_time:.4f} ms")
    print(f"{'RK45 (Dormand-Prince)':<25} | {'Adaptive':<15} | {rk45_time:.4f} ms")