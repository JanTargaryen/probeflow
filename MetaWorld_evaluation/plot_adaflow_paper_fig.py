import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors
import cv2
import os

def get_interpolated_points(u, v, steps_n, is_baseline=False):
    u_out, v_out, c_out = [], [], []
    for i in range(len(u) - 1):
        u_start, v_start = u[i], v[i]
        u_end, v_end = u[i+1], v[i+1]
        
        num_points = 50 if is_baseline else int(steps_n[i])
        t_vals = np.linspace(0, 1, num_points, endpoint=False)
        for t in t_vals:
            u_out.append(u_start + t * (u_end - u_start))
            v_out.append(v_start + t * (v_end - v_start))
            c_out.append(50 if is_baseline else steps_n[i]) 

    u_out.append(u[-1])
    v_out.append(v[-1])
    c_out.append(50 if is_baseline else steps_n[-1])

    return np.array(u_out), np.array(v_out), np.array(c_out)

def plot_adaflow_paper_fig(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    bg_img = cv2.cvtColor(data['bg_img'], cv2.COLOR_BGR2RGB)
    
    u_raw = data['uv'][:, 0]
    v_raw = data['uv'][:, 1]
    steps_raw = data['steps']
    
    u_base, v_base, _ = get_interpolated_points(u_raw, v_raw, steps_raw, is_baseline=True)
    u_ada, v_ada, c_ada = get_interpolated_points(u_raw, v_raw, steps_raw, is_baseline=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    line_effects = [path_effects.Stroke(linewidth=3.5, foreground='black'), path_effects.Normal()]
    
    baseline_color = 'dimgray' 
    
    # =====================================
    # Left: Baseline (Fixed-Step)
    # =====================================
    ax1.imshow(bg_img)
    ax1.axis('off')
    
    line1, = ax1.plot(u_raw, v_raw, color='white', linewidth=2.5, alpha=0.4, zorder=1)
    line1.set_path_effects(line_effects)
    
    ax1.scatter(u_base, v_base, color=baseline_color, s=20, 
                edgecolors='black', linewidths=0.3, zorder=2, alpha=0.85)
                
    # 创建左侧专属的"纯色" Colorbar
    cmap_base = mcolors.ListedColormap([baseline_color])
    norm_base = mcolors.Normalize(vmin=49, vmax=51) 
    sm_base = plt.cm.ScalarMappable(cmap=cmap_base, norm=norm_base)
    sm_base.set_array([])
    cbar1 = fig.colorbar(sm_base, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_ticks([50]) # 只显示 50 这个刻度
    
    # =====================================
    # Right: Ours (AdaFlow)
    # =====================================
    ax2.imshow(bg_img)
    ax2.axis('off')
    
    line2, = ax2.plot(u_raw, v_raw, color='white', linewidth=2.5, alpha=0.4, zorder=1)
    line2.set_path_effects(line_effects)
    
    point_sizes = 20 + (c_ada - c_ada.min()) * 4
    GLOBAL_VMIN = 2
    GLOBAL_VMAX = 8 
    
    sc = ax2.scatter(u_ada, v_ada, c=c_ada, cmap='coolwarm', s=point_sizes, 
                     edgecolors='black', linewidths=0.6, zorder=2, alpha=0.95,
                     vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX) 
    
    # 创建右侧专属的"渐变色" Colorbar
    cbar2 = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_ticks(np.arange(GLOBAL_VMIN, GLOBAL_VMAX + 1, 2))

    plt.tight_layout() # 自动排版，防止标签重叠
    plt.savefig("adaflow_trajectory_comparison.pdf", bbox_inches='tight', transparent=True)
    plt.savefig("adaflow_trajectory_comparison.png", bbox_inches='tight', transparent=True)
    print("Success: MetaWorld figure saved.")

if __name__ == "__main__":
    target_npy_path = "/mnt/data_ssd/zhoufang/code/evo-fast/MetaWorld_evaluation/traj_data_for_plot/traj_task04_button-press-topdown-v3_ep0.npy" 
    if os.path.exists(target_npy_path):
        plot_adaflow_paper_fig(target_npy_path)
    else:
        print(f"File not found: {target_npy_path}")