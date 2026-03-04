import torch
import numpy as np
from scipy.integrate import RK45

def test_rk45_implementation():
    # 1. Define a non-linear ODE: dy/dt = sin(t) * y
    def v_fn_np(t, y):
        return np.sin(t) * y

    def v_fn_pt(y, t):
        return torch.sin(t) * y

    t0, t_bound = 0.0, 5.0
    y0_val = 1.0
    rtol, atol = 1e-3, 1e-5

    # 2. Run SciPy RK45 (Ground Truth)
    solver_sp = RK45(v_fn_np, t0, [y0_val], t_bound, rtol=rtol, atol=atol)
    t_sp, y_sp = [t0], [y0_val]
    while solver_sp.status == 'running':
        solver_sp.step()
        t_sp.append(solver_sp.t)
        y_sp.append(solver_sp.y[0])

    # 3. Run Custom PyTorch RK45
    t = torch.tensor(t0, dtype=torch.float64)
    dt = torch.tensor(0.1, dtype=torch.float64)
    action = torch.tensor([y0_val], dtype=torch.float64)
    
    # DP54 Coefficients
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
    e1, e3, e4, e5, e6, e7 = 71/57600, -71/16695, 71/1920, -17253/339200, 22/525, -1/40

    t_pt, y_pt = [t0], [y0_val]

    while t < t_bound - 1e-5:
        if t + dt > t_bound:
            dt = t_bound - t
            
        k1 = v_fn_pt(action, t)
        k2 = v_fn_pt(action + dt * (a21 * k1), t + 0.2 * dt)
        k3 = v_fn_pt(action + dt * (a31 * k1 + a32 * k2), t + 0.3 * dt)
        k4 = v_fn_pt(action + dt * (a41 * k1 + a42 * k2 + a43 * k3), t + 0.8 * dt)
        k5 = v_fn_pt(action + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), t + (8/9) * dt)
        k6 = v_fn_pt(action + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), t + dt)
        
        action_next = action + dt * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        k7 = v_fn_pt(action_next, t + dt)
        
        err_vec = dt * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7)
        scale = atol + rtol * torch.max(torch.abs(action), torch.abs(action_next))
        err = torch.max(torch.abs(err_vec) / scale).item()
        
        dt_next = dt * max(0.2, min(5.0, 0.9 * (err + 1e-8)**(-0.2)))
        
        if err <= 1.0:
            action = action_next
            t += dt
            t_pt.append(t.item())
            y_pt.append(action.item())
        dt = dt_next

    # 4. Compare final results
    scipy_final = y_sp[-1]
    pytorch_final = y_pt[-1]
    abs_diff = abs(scipy_final - pytorch_final)
    
    print("-" * 40)
    print(f"SciPy RK45 Final Value:   {scipy_final:.8f}")
    print(f"PyTorch RK45 Final Value: {pytorch_final:.8f}")
    print(f"Absolute Difference:      {abs_diff:.8e}")
    print("-" * 40)
    
    if abs_diff < 1e-5:
        print("Verification Passed: PyTorch implementation matches SciPy RK45.")
    else:
        print("Verification Failed: Significant divergence detected.")

if __name__ == "__main__":
    test_rk45_implementation()