import matplotlib.pyplot as plt
import numpy as np

probe_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
success_rates = [0.0, 1.8, 1.4, 4.6, 83.2, 81.2, 85.8, 84.8, 81.0]
avg_steps = [2.0, 2.0, 2.0, 2.03, 2.6, 4.22, 6.5, 13.03, 20.0]

# 设置全局字体大小和样式，使其符合 IEEE 论文规范
plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

fig, ax1 = plt.subplots(figsize=(8, 5))

# 左侧 Y 轴：Success Rate (百分比)
color1 = '#1f77b4' # 深蓝色
ax1.set_xlabel(r'Lookahead Probe Horizon ($\Delta t_{probe}$)', fontsize=16)
ax1.set_ylabel('Success Rate (%)', color=color1, fontsize=16)
line1, = ax1.plot(probe_steps, success_rates, marker='o', color=color1, linewidth=2.5, markersize=8, label='Success Rate')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(-5, 100)
ax1.grid(True, linestyle='--', alpha=0.6)

# 实例化一个共享 x 轴的右侧 Y 轴：Average Steps
ax2 = ax1.twinx()  
color2 = '#ff7f0e' # 亮橙色
ax2.set_ylabel('Average Inference Steps ($N$)', color=color2, fontsize=16)
line2, = ax2.plot(probe_steps, avg_steps, marker='s', color=color2, linewidth=2.5, markersize=8, linestyle='--', label='Average Steps')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 15)

# 合并图例
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(0.05, 0.65), framealpha=0.9)

plt.title('Ablation on Lookahead Probe Horizon', fontsize=16, pad=15)
plt.tight_layout()

# 保存为 PDF 格式（矢量图，放大不失真，IEEE 强烈推荐）
plt.savefig('probe_horizon_ablation.pdf', dpi=300, bbox_inches='tight')
plt.show()