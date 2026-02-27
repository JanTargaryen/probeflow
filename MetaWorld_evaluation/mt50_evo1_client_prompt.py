import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import cv2
import os

def get_interpolated_points(u, v, steps_n, is_baseline=False):
    # Simulate sampling density without data smoothing
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    plt.subplots_adjust(wspace=0.03) 
    line_effects = [path_effects.Stroke(linewidth=3.5, foreground='black'), path_effects.Normal()]
    
    baseline_color = 'dimgray' 
    
    # Left: Baseline (Fixed-Step)
    ax1.imshow(bg_img)
    ax1.axis('off')
    ax1.set_title("Baseline: Fixed-Step Euler ($N=50$)", fontsize=16, fontweight='bold', pad=15)
    
    line1, = ax1.plot(u_raw, v_raw, color='white', linewidth=2.5, alpha=0.4, zorder=1)
    line1.set_path_effects(line_effects)
    
    ax1.scatter(u_base, v_base, color=baseline_color, s=20, 
                edgecolors='black', linewidths=0.3, zorder=2, alpha=0.85)
    ax1.text(0.05, 0.05, '$N=50$ (Constant)', 
             transform=ax1.transAxes, 
             fontsize=14, fontweight='bold', color='white', 
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='dimgray', edgecolor='black', alpha=0.85))
    # Right: Ours (AdaFlow)
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
    
    # Render Colorbar exclusively for AdaFlow
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7]) 
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_ticks(np.arange(GLOBAL_VMIN, GLOBAL_VMAX + 1, 2))
    cbar.set_label('Allocated ODE Steps ($N$) [AdaFlow Only]', fontsize=14, fontweight='bold', rotation=270, labelpad=20)

    plt.savefig("adaflow_trajectory_comparison.pdf", bbox_inches='tight', transparent=True)
    plt.savefig("adaflow_trajectory_comparison.png", bbox_inches='tight', transparent=True)
    print("Success: MetaWorld figure saved.")

if __name__ == "__main__":
    target_npy_path = "/mnt/data_ssd/zhoufang/code/evo-fast/MetaWorld_evaluation/traj_data_for_plot/traj_task00_nut-assembly-v3_ep0.npy" 
    if os.path.exists(target_npy_path):
        plot_adaflow_paper_fig(target_npy_path)
    else:
        print(f"File not found: {target_npy_path}")