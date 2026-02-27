import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors
import os

def distribute_points_along_segment(uv_path, num_points):
    uv_path = np.array(uv_path)
    if len(uv_path) <= 1:
        return np.tile(uv_path[-1] if len(uv_path)==1 else [0,0], (num_points, 1))
    
    diffs = np.diff(uv_path, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_dist = cum_dists[-1]
    
    if total_dist == 0:
        return np.tile(uv_path[0], (num_points, 1))
    
    target_dists = np.linspace(0, total_dist, num_points, endpoint=False)
    u_interp = np.interp(target_dists, cum_dists, uv_path[:, 0])
    v_interp = np.interp(target_dists, cum_dists, uv_path[:, 1])
    
    return np.column_stack((u_interp, v_interp))

def plot_adaflow_paper_fig(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    bg_img = data['bg_img']
    trajectory_data = data['trajectory_data']
    
    all_u, all_v = [], []
    base_u, base_v, base_c = [], [], []
    ada_u, ada_v, ada_c = [], [], []
    
    for seg in trajectory_data:
        uv_path = np.array(seg['uv_path'])
        if len(uv_path) == 0: 
            continue
        n_steps = int(seg['n_steps'])
        
        all_u.extend(uv_path[:, 0])
        all_v.extend(uv_path[:, 1])
        
        pts_base = distribute_points_along_segment(uv_path, 50)
        base_u.extend(pts_base[:, 0])
        base_v.extend(pts_base[:, 1])
        base_c.extend([50] * 50)
        
        pts_ada = distribute_points_along_segment(uv_path, n_steps)
        ada_u.extend(pts_ada[:, 0])
        ada_v.extend(pts_ada[:, 1])
        ada_c.extend([n_steps] * n_steps)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    line_effects = [path_effects.Stroke(linewidth=3.5, foreground='black'), path_effects.Normal()]
    baseline_color = 'dimgray' 
    
    ax1.imshow(bg_img)
    ax1.axis('off')
    ax1.set_title("Baseline: Fixed-Step Euler ($N=50$)", fontsize=16, fontweight='bold', pad=15)
    
    ax1.plot(all_u, all_v, color='white', linewidth=2.5, alpha=0.4, zorder=1, path_effects=line_effects)
    ax1.scatter(base_u, base_v, color=baseline_color, s=20, 
                edgecolors='black', linewidths=0.3, zorder=2, alpha=0.85)
                
    cmap_base = mcolors.ListedColormap([baseline_color])
    norm_base = mcolors.Normalize(vmin=49, vmax=51)
    sm_base = plt.cm.ScalarMappable(cmap=cmap_base, norm=norm_base)
    sm_base.set_array([])
    cbar1 = fig.colorbar(sm_base, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_ticks([50])
    cbar1.set_label('Constant ODE Steps ($N$)', fontsize=14, fontweight='bold', rotation=270, labelpad=20)
    
    ax2.imshow(bg_img)
    ax2.axis('off')
    ax2.set_title("Ours: AdaFlow (Linearity-Aware Adaptive Solver)", fontsize=16, fontweight='bold', pad=15)
    
    ax2.plot(all_u, all_v, color='white', linewidth=2.5, alpha=0.4, zorder=1, path_effects=line_effects)
    
    c_ada = np.array(ada_c)
    point_sizes = 20 + (c_ada - c_ada.min()) * 4 if len(c_ada) > 0 else 20
    GLOBAL_VMIN, GLOBAL_VMAX = 2, 8 
    
    sc = ax2.scatter(ada_u, ada_v, c=ada_c, cmap='coolwarm', s=point_sizes, 
                     edgecolors='black', linewidths=0.6, zorder=2, alpha=0.95,
                     vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX) 
    
    cbar2 = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_ticks(np.arange(GLOBAL_VMIN, GLOBAL_VMAX + 1, 2))
    cbar2.set_label('Adaptive ODE Steps ($N$)', fontsize=14, fontweight='bold', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig("libero_adaflow_trajectory.pdf", bbox_inches='tight', transparent=True)
    plt.savefig("libero_adaflow_trajectory.png", bbox_inches='tight', transparent=True)
    print("Success: Dense physical LIBERO figure saved.")

if __name__ == "__main__":
    target_npy_path = "/mnt/data_ssd/zhoufang/code/evo-fast/LIBERO_evaluation/traj_data_for_plot_libero/traj_task00_libero_spatial_ep0.npy" 
    if os.path.exists(target_npy_path):
        plot_adaflow_paper_fig(target_npy_path)
    else:
        print(f"File not found: {target_npy_path}")