#!/usr/bin/env python3
"""Generate join_between_progress.png from join_between_results.tsv."""
import sys
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available, skipping plot")
    sys.exit(0)

tsv_path = os.path.join(os.path.dirname(__file__), '..', 'join_between_results.tsv')
out_path = os.path.join(os.path.dirname(__file__), 'join_between_progress.png')

rows = []
with open(tsv_path) as f:
    header = f.readline().strip().split('\t')
    for line in f:
        parts = line.strip().split('\t')
        if not parts or len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        rows.append(row)

if not rows:
    print("No data rows found")
    sys.exit(0)

color_map = {'keep': 'green', 'discard': 'red', 'crash': 'orange'}

indices = list(range(len(rows)))
pl_times = []
pl_mems = []
np_times = []
np_mems = []
colors = []
labels = []

for row in rows:
    try:
        pl_times.append(float(row['pl_time']))
        pl_mems.append(float(row['pl_mem_mb']))
        np_times.append(float(row['np_time']))
        np_mems.append(float(row.get('np_mem_mb', 0)))
        colors.append(color_map.get(row['status'], 'blue'))
        labels.append(row['description'][:30])
    except (ValueError, KeyError):
        pass

best_fitness = min(float(r['fitness']) for r in rows if r.get('fitness') not in ('', '-', 'N/A'))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f'join_between optimization progress\nBest fitness: {best_fitness:.3f}', fontsize=14)

ax1.scatter(indices, pl_times, c=colors, s=100, zorder=3)
ax1.plot(indices, pl_times, c='gray', alpha=0.5, zorder=2)
if np_times:
    ax1.axhline(y=np_times[0], color='blue', linestyle='--', alpha=0.7, label=f'np_time={np_times[0]:.4f}s')
ax1.set_ylabel('pl_time (s)')
ax1.set_yscale('log' if max(pl_times) / min(pl_times) > 10 else 'linear')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.scatter(indices, pl_mems, c=colors, s=100, zorder=3)
ax2.plot(indices, pl_mems, c='gray', alpha=0.5, zorder=2)
if np_mems:
    ax2.axhline(y=np_mems[0], color='blue', linestyle='--', alpha=0.7, label=f'np_mem={np_mems[0]:.1f}MB')
ax2.set_ylabel('pl_mem_mb (MB)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add status legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='keep'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='discard'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='crash'),
]
ax1.legend(handles=legend_elements + ax1.get_legend_handles_labels()[0], loc='upper right')

plt.xticks(indices, labels, rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(out_path, dpi=100, bbox_inches='tight')
print(f"Plot saved to {out_path}")
