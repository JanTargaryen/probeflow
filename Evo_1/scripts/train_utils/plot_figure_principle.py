import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc

def draw_vector(ax, start, direction, color, label=None, label_offset=(0,0), linestyle='-', alpha=1.0):
    end = start + direction
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=35, 
                            color=color, linestyle=linestyle, alpha=alpha, lw=3.0)
    ax.add_patch(arrow)
    if label:
        ax.text(start[0] + direction[0]/2 + label_offset[0], 
                start[1] + direction[1]/2 + label_offset[1], 
                label, fontsize=28, color='black', weight='bold', math_fontfamily='cm')
    return end

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    # ==========================================
    # Left Panel: Linear Region
    # ==========================================
    ax1.set_title("Linear Region", fontsize=34, fontweight='bold', pad=20)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_xticks([])
    ax1.set_yticks([])

    x = np.linspace(0, 10, 100)
    for offset in [-3, -1.5, 0, 1.5, 3]:
        ax1.plot(x, 0.5 * x + 2.5 + offset, color='gray', linestyle='--', alpha=0.4, lw=1.5)
    
    ax1.plot(x, 0.5 * x + 2.5, color='black', lw=2.5)

    x0 = np.array([2.0, 3.5])
    v_start_dir = np.array([1.0, 0.5])
    v_start = (v_start_dir / np.linalg.norm(v_start_dir)) * 2.5
    dt = 1.2
    x_probe = x0 + v_start * dt

    ax1.plot(*x0, 'ko', markersize=12)
    ax1.text(x0[0]+0.1, x0[1]-1.1, r'$\boldsymbol{x}_0$', fontsize=28)
    draw_vector(ax1, x0, v_start, 'red', r'$\boldsymbol{v}_{start}$', (0.1, -1.0))

    ax1.plot(*x_probe, 'ko', markersize=12)
    # --- MODIFICATION START ---
    ax1.text(x_probe[0]+0.1, x_probe[1]-1.3, r'$\boldsymbol{x}_{probe}$', fontsize=28)
    
    v_probe = v_start * 0.9 
    draw_vector(ax1, x_probe, v_probe, 'blue', r'$\boldsymbol{v}_{probe}$', (0.1, 0.8))
    
    ax1.text(x_probe[0] + 0.5, x_probe[1] - 4.2, r'$\mathcal{S} \approx 1$', fontsize=34)
    # --- MODIFICATION END ---

    # ==========================================
    # Right Panel: Curved Region
    # ==========================================
    ax2.set_title("Curved Region", fontsize=34, fontweight='bold', pad=20)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_xticks([])
    ax2.set_yticks([])

    def curve_func(x):
        return 1.8 * np.sin(x / 2.2) + 4.5
    
    def curve_deriv(x):
        return (1.8 / 2.2) * np.cos(x / 2.2)

    for offset in [-4, -2, 0, 2, 4]:
        ax2.plot(x, curve_func(x) + offset, color='gray', linestyle='--', alpha=0.4, lw=1.5)
    ax2.plot(x, curve_func(x), color='black', lw=2.5)

    x0_val = 2.0
    x0 = np.array([x0_val, curve_func(x0_val)])
    
    slope_0 = curve_deriv(x0_val)
    v_start_dir = np.array([1.0, slope_0])
    v_start = (v_start_dir / np.linalg.norm(v_start_dir)) * 2.5 

    x_probe = x0 + v_start * dt

    ax2.plot(*x0, 'ko', markersize=12)
    ax2.text(x0[0]+0.1, x0[1]-1.1, r'$\boldsymbol{x}_0$', fontsize=28)
    draw_vector(ax2, x0, v_start, 'red', r'$\boldsymbol{v}_{start}$', (-1.5, 0.6))

    ax2.plot(*x_probe, 'ko', markersize=12)
    # --- MODIFICATION START ---
    ax2.text(x_probe[0]-0.5, x_probe[1]+0.6, r'$\boldsymbol{x}_{probe}$', fontsize=28)
    # --- MODIFICATION END ---

    slope_probe = curve_deriv(x_probe[0])
    v_probe_dir = np.array([1.0, slope_probe])
    v_probe = (v_probe_dir / np.linalg.norm(v_probe_dir)) * 2.5
    draw_vector(ax2, x_probe, v_probe, 'blue', r'$\boldsymbol{v}_{probe}$', (0.2, -1))

    draw_vector(ax2, x_probe, v_start, 'red', linestyle='--', alpha=0.4)

    angle1 = np.degrees(np.arctan2(v_start[1], v_start[0]))
    angle2 = np.degrees(np.arctan2(v_probe[1], v_probe[0]))
    
    theta1, theta2 = min(angle1, angle2), max(angle1, angle2)
    if theta2 - theta1 > 180:
        theta1, theta2 = theta2, theta1 + 360

    arc = Arc(x_probe, 1.5, 1.5, angle=0, theta1=theta1, theta2=theta2, color='black', lw=2.5)
    ax2.add_patch(arc)
    
    text_r = 1.6 
    mid_angle = np.radians((theta1 + theta2) / 2)
    ax2.text(x_probe[0] + text_r * np.cos(mid_angle) - 0.2, 
             x_probe[1] + text_r * np.sin(mid_angle) - 0.2, 
             r'$\theta$', fontsize=28)

    # --- MODIFICATION START ---
    ax2.text(x_probe[0] - 1.5, x_probe[1] - 5.2, r'$\mathcal{S} = \cos \theta \ll 1$', fontsize=34)
    # --- MODIFICATION END ---

    plt.tight_layout()
    plt.savefig('probeflow_mechanism_acute.pdf', format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()