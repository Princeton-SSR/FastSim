#!/usr/bin/python
"""Plots simulation data from a sweep of n_magnitude_leds values across all formations.

Reads the manifest written by sweep_n_magnitude_leds.py (mapping each formation
and swept n_magnitude_leds value to the logfile prefix it produced) and overlays
runs in a grid: same-plane formations (F1S/F2S/F3S) on top as trajectory+distance
row pairs, different-plane formations (F1B/F2B/F3B) below, colored from
light (low noise) to dark (high noise).

Usage:
    python plot_sweep_n_magnitude_leds.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from fishfood.exp_formations import CONFIGS

MANIFEST_PATH = './logfiles/sweep_n_magnitude_leds_manifest.json'

# Sequential blue ramp, light->dark, one step per swept value (see dataviz skill).
SEQUENTIAL_STEPS = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']

with open(MANIFEST_PATH, 'r') as f:
    manifest = json.load(f)

sweep_values = sorted(next(iter(manifest.values())).keys(), key=float)
colors = SEQUENTIAL_STEPS if len(sweep_values) <= len(SEQUENTIAL_STEPS) else \
    plt.cm.Blues(np.linspace(0.35, 0.95, len(sweep_values)))

plt.rcParams['svg.fonttype'] = 'none'  # Set font to be recognizable in SVG

# Row layout: trajectory row then distance row, once for each formation group
GROUPS = [['F1S', 'F2S', 'F3S'], ['F1B', 'F2B', 'F3B']]
n_cols = len(GROUPS[0])
n_row_pairs = len(GROUPS)
fig, axes = plt.subplots(2 * n_row_pairs, n_cols, constrained_layout=True, figsize=(4 * n_cols, 3.4 * 2 * n_row_pairs))
fig.get_layout_engine().set(hspace=0.08)

for group_idx, group in enumerate(GROUPS):
    traj_row = axes[2 * group_idx]
    dist_row = axes[2 * group_idx + 1]

    for col, formation in enumerate(group):
        ax1 = traj_row[col]
        ax3 = dist_row[col]
        arena = None

        for value, color in zip(sweep_values, colors):
            filename = manifest[formation][value]
            try:
                data = np.loadtxt('./logfiles/{}_data.txt'.format(filename), delimiter=',')
                with open('./logfiles/{}_meta.txt'.format(filename), 'r') as f:
                    meta = json.loads(f.read())
            except OSError:
                data = np.loadtxt('./logfiles/{}_0_data.txt'.format(filename), delimiter=',')
                with open('./logfiles/{}_0_meta.txt'.format(filename), 'r') as f:
                    meta = json.loads(f.read())

            clock_freq = meta['Clock frequency [Hz]']
            arena = meta['Arena [mm]']
            no_leader = meta['Number of leaders']
            fishes = int(data.shape[1] / 8)
            timesteps = data.shape[0]
            t = np.arange(0, timesteps, 1) / clock_freq

            label = 'n={}'.format(value)

            # Global trajectory
            for ii in range(no_leader, fishes):
                ax1.plot(data[:, 4*ii], data[:, 4*ii+1], color=color, linewidth=1, label=label)

            # Distance (in leader/follower frame)
            x0, y0, phi0 = data[:, 0], data[:, 1], data[:, 3]
            for ii in range(no_leader, fishes):
                x = x0 - data[:, 4*ii]
                y = y0 - data[:, 4*ii+1]
                z_rel = data[:, 2] - data[:, 4*ii+2]
                x_rel = x * np.cos(phi0) + y * np.sin(phi0)
                y_rel = x * -np.sin(phi0) + y * np.cos(phi0)

                dist = np.linalg.norm((x_rel, y_rel, z_rel), axis=0)
                ax3.plot(t, dist, color=color, linewidth=1.5, label=label)

        # Configure trajectory subplot
        ax1.set_aspect('equal')
        ax1.plot(arena[0]/2*np.cos(np.linspace(0, 2*np.pi, 500)),
                  arena[0]/2*np.sin(np.linspace(0, 2*np.pi, 500)), color='gray', linewidth=0.5)
        ax1.set_xlim([-arena[0]/2-100, arena[0]/2+100])
        ax1.set_ylim([-arena[0]/2-100, arena[0]/2+100])
        ax1.axis('off')
        ax1.set_title(formation, fontsize=13)

        # Target reference line, from this formation's config (safe_distance, approach_distance, distance, angle, pitch_range)
        _, _, target_distance, _, _ = CONFIGS[formation]

        # Configure distance subplot
        ax3.set(xlabel='Time (s)')
        ax3.set_ylim([0, 2000])
        ax3.axhline(target_distance, color='gray', linestyle='--', linewidth=1, zorder=0)

    traj_row[0].text(-0.3, 0.5, 'Global\ntrajectory', fontsize=12, ha='right', va='center', transform=traj_row[0].transAxes)
    dist_row[0].set_ylabel('Distance (mm)', fontsize=12)

# Single legend for the whole figure (one entry per swept n_magnitude_leds value)
handles, labels = axes[0, 0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='outside upper center', ncol=len(sweep_values),
                     frameon=False, title='n_magnitude_leds', fontsize=14, title_fontsize=15)
for line in legend.get_lines():
    line.set_linewidth(3)

if not os.path.exists('./logfigs'):
    os.makedirs('./logfigs')

fig.savefig('./logfigs/sweep_n_magnitude_leds_comparison.png', dpi=300, bbox_inches='tight', transparent=True)
fig.savefig('./logfigs/sweep_n_magnitude_leds_comparison.svg', bbox_inches='tight', transparent=True)
print('Saved comparison plot to ./logfigs/sweep_n_magnitude_leds_comparison.{png,svg}')

plt.show()
