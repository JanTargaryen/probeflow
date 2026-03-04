import numpy as np
import matplotlib.pyplot as plt

# Set font for IEEE style
plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# --- Plot 1: Linear Flow (Straight Path) ---
ax1 = axs[0]
Y, X = np.mgrid[-2:2:100j, -2:2:100j]
# Constant vector field
U1 = np.ones_like(X) * 1.5
V1 = np.ones_like(Y) * 0.5

# Streamplot
ax1.streamplot(X, Y, U1, V1, color='lightgray', linewidth=1, density=1.0)

# Points
x0_1, y0_1 = -1.0, -0.5
dt = 0.8 # probe step
v_start_x1, v_start_y1 = 1.5, 0.5
norm1 = np.sqrt(v_start_x1**2 + v_start_y1**2)
vx1, vy1 = v_start_x1/norm1, v_start_y1/norm1

x_probe_1 = x0_1 + vx1 * dt * 1.5
y_probe_1 = y0_1 + vy1 * dt * 1.5

ax1.plot([x0_1, x_probe_1], [y0_1, y_probe_1], 'k--', linewidth=2, alpha=0.7)

# Quivers
ax1.quiver(x0_1, y0_1, vx1, vy1, color='red', scale=4, zorder=5, width=0.015)
ax1.quiver(x_probe_1, y_probe_1, vx1, vy1, color='blue', scale=4, zorder=5, width=0.015)

# Scatters
ax1.scatter([x0_1], [y0_1], color='black', s=60, zorder=6)
ax1.scatter([x_probe_1], [y_probe_1], color='black', s=60, zorder=6)

# Annotations
ax1.text(x0_1 - 0.2, y0_1 + 0.1, r'$\mathbf{x}_0$', fontsize=16, fontweight='bold')
ax1.text(x_probe_1 - 0.2, y_probe_1 + 0.2, r'$\mathbf{x}_{probe}$', fontsize=16, fontweight='bold')
ax1.text(x0_1 + 0.3, y0_1 - 0.3, r'$\mathbf{v}_{start}$', color='red', fontsize=16, fontweight='bold')
ax1.text(x_probe_1 + 0.3, y_probe_1 - 0.3, r'$\mathbf{v}_{probe}$', color='blue', fontsize=16, fontweight='bold')

ax1.set_xlim(-2, 2)
ax1.set_ylim(-1.5, 1.5)
ax1.set_title('Linear Flow Region\n(High Cosine Similarity)', pad=15)
ax1.set_xticks([])
ax1.set_yticks([])


# --- Plot 2: Curved Flow (Non-linear Path) ---
ax2 = axs[1]
# Rotational/Curved vector field
U2 = -Y
V2 = X + 1.0

# Streamplot
ax2.streamplot(X, Y, U2, V2, color='lightgray', linewidth=1, density=1.0)

# Points
x0_2, y0_2 = -0.5, -0.5
v_start_x2, v_start_y2 = -y0_2, x0_2 + 1.0
norm2_start = np.sqrt(v_start_x2**2 + v_start_y2**2)
vx2_start, vy2_start = v_start_x2/norm2_start, v_start_y2/norm2_start

x_probe_2 = x0_2 + vx2_start * dt * 1.5
y_probe_2 = y0_2 + vy2_start * dt * 1.5
v_probe_x2, v_probe_y2 = -y_probe_2, x_probe_2 + 1.0
norm2_probe = np.sqrt(v_probe_x2**2 + v_probe_y2**2)
vx2_probe, vy2_probe = v_probe_x2/norm2_probe, v_probe_y2/norm2_probe

ax2.plot([x0_2, x_probe_2], [y0_2, y_probe_2], 'k--', linewidth=2, alpha=0.7) 

# Quivers
ax2.quiver(x0_2, y0_2, vx2_start, vy2_start, color='red', scale=4, zorder=5, width=0.015)
ax2.quiver(x_probe_2, y_probe_2, vx2_probe, vy2_probe, color='blue', scale=4, zorder=5, width=0.015)

# Scatters
ax2.scatter([x0_2], [y0_2], color='black', s=60, zorder=6)
ax2.scatter([x_probe_2], [y_probe_2], color='black', s=60, zorder=6)

# Annotations
ax2.text(x0_2 - 0.3, y0_2 - 0.2, r'$\mathbf{x}_0$', fontsize=16, fontweight='bold')
ax2.text(x_probe_2 - 0.1, y_probe_2 - 0.3, r'$\mathbf{x}_{probe}$', fontsize=16, fontweight='bold')
ax2.text(x0_2 + 0.1, y0_2 + 0.4, r'$\mathbf{v}_{start}$', color='red', fontsize=16, fontweight='bold')
ax2.text(x_probe_2 - 0.6, y_probe_2 + 0.2, r'$\mathbf{v}_{probe}$', color='blue', fontsize=16, fontweight='bold')

ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.0, 2.0)
ax2.set_title('Curved Flow Region\n(Low Cosine Similarity)', pad=15)
ax2.set_xticks([])
ax2.set_yticks([])

plt.tight_layout()
plt.savefig('method_principle.pdf', dpi=300, bbox_inches='tight')