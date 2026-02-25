import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import cv2
import os

def get_interpolated_points(u, v, steps_n, is_baseline=False):
    """通过真实的物理打点模拟采样密度（无任何数据平滑，保留原汁原味）"""
    u_out, v_out, c_out = [], [], []
    
    for i in range(len(u) - 1):
        u_start, v_start = u[i], v[i]
        u_end, v_end = u[i+1], v[i+1]
        
        if is_baseline:
            num_points = 25  # Baseline 密集采样
        else:
            num_points = int(steps_n[i]) # AdaFlow 动态步数
            
        t_vals = np.linspace(0, 1, num_points, endpoint=False)
        for t in t_vals:
            u_out.append(u_start + t * (u_end - u_start))
            v_out.append(v_start + t * (v_end - v_start))
            c_out.append(steps_n[i]) 

    u_out.append(u[-1])
    v_out.append(v[-1])
    c_out.append(steps_n[-1])

    return np.array(u_out), np.array(v_out), np.array(c_out)

def plot_adaflow_paper_fig(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    bg_img = cv2.cvtColor(data['bg_img'], cv2.COLOR_BGR2RGB)
    
    u_raw = data['uv'][:, 0]
    v_raw = data['uv'][:, 1]
    steps_raw = data['steps']
    
    # 获取模拟打点
    u_base, v_base, _ = get_interpolated_points(u_raw, v_raw, steps_raw, is_baseline=True)
    u_ada, v_ada, c_ada = get_interpolated_points(u_raw, v_raw, steps_raw, is_baseline=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    plt.subplots_adjust(wspace=0.03) 
    line_effects = [path_effects.Stroke(linewidth=3.5, foreground='black'), path_effects.Normal()]
    
    # =====================================
    # 左图：Baseline (Fixed-Step)
    # =====================================
    ax1.imshow(bg_img)
    ax1.axis('off')
    ax1.set_title("Baseline: Fixed-Step Euler ($N=50$)", fontsize=16, fontweight='bold', pad=15)
    
    line1, = ax1.plot(u_raw, v_raw, color='white', linewidth=2.5, alpha=0.4, zorder=1)
    line1.set_path_effects(line_effects)
    

    ax2.imshow(bg_img)
    ax2.axis('off')
    ax2.set_title("Ours: AdaFlow (Linearity-Aware Quantization)", fontsize=16, fontweight='bold', pad=15)
    
    line2, = ax2.plot(u_raw, v_raw, color='white', linewidth=2.5, alpha=0.4, zorder=1)
    line2.set_path_effects(line_effects)
    
    point_sizes = 20 + (c_ada - c_ada.min()) * 4
    
    GLOBAL_VMIN = 2
    GLOBAL_VMAX = 8 
    
    sc = ax2.scatter(u_ada, v_ada, c=c_ada, cmap='coolwarm', s=point_sizes, 
                     edgecolors='black', linewidths=0.6, zorder=2, alpha=0.95,
                     vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX) 
    
    # 添加 Colorbar
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7]) 
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_ticks(np.arange(GLOBAL_VMIN, GLOBAL_VMAX + 1, 2))
    cbar.set_label('Allocated ODE Steps ($N$)', fontsize=14, fontweight='bold', rotation=270, labelpad=20)

    plt.savefig("adaflow_trajectory_comparison.pdf", bbox_inches='tight', transparent=True)
    plt.savefig("adaflow_trajectory_comparison.png", bbox_inches='tight', transparent=True)
    print("Success! Clean academic figure saved.")

if __name__ == "__main__":
    target_npy_path = "/mnt/data_ssd/zhoufang/code/evo-fast/MetaWorld_evaluation/traj_data_for_plot/traj_task04_button-press-topdown-v3_ep0.npy" 
    if os.path.exists(target_npy_path):
        plot_adaflow_paper_fig(target_npy_path)