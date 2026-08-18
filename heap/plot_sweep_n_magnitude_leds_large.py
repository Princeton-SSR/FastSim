#!/usr/bin/python
"""Plots simulation data from a sweep of n_magnitude_leds values for the exp8 ("n8") experiment.

Reads the manifest written by sweep_n_magnitude_leds_large.py (mapping each
swept n_magnitude_leds value to the logfile prefix it produced) and plots each
noise level in its own column (trajectory+distance row pair), with the 7
followers colored individually so they can be told apart within a panel and
tracked across panels.

Usage:
    python plot_sweep_n_magnitude_leds_large.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from fishfood.exp8 import combinations

MANIFEST_PATH = './logfiles/sweep_n_magnitude_leds_large_manifest.json'

# Categorical palette, slots 1-7, fixed order (see dataviz skill) - one color per follower.
FOLLOWER_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']

with open(MANIFEST_PATH, 'r') as f:
    manifest = json.load(f)

formation = next(iter(manifest.keys()))
sweep_values = sorted(manifest[formation].keys(), key=float)

plt.rcParams['svg.fonttype'] = 'none'  # Set font to be recognizable in SVG

# Every follower in exp8's combinations targets the same distance from the leader.
target_distance = combinations[0, 1]

n_cols = len(sweep_values)
fig, axes = plt.subplots(2, n_cols, constrained_layout=True, figsize=(4 * n_cols, 6.8))
traj_row, dist_row = axes[0], axes[1]

for col, value in enumerate(sweep_values):
    ax1 = traj_row[col]
    ax3 = dist_row[col]

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

    # Global trajectory, one color per follower
    for i, ii in enumerate(range(no_leader, fishes)):
        color = FOLLOWER_COLORS[i % len(FOLLOWER_COLORS)]
        label = 'Fish {}'.format(ii) if col == 0 else None
        ax1.plot(data[:, 4*ii], data[:, 4*ii+1], color=color, linewidth=1, label=label)

    # Distance (in leader/follower frame), same per-follower colors
    x0, y0, phi0 = data[:, 0], data[:, 1], data[:, 3]
    for i, ii in enumerate(range(no_leader, fishes)):
        color = FOLLOWER_COLORS[i % len(FOLLOWER_COLORS)]
        x = x0 - data[:, 4*ii]
        y = y0 - data[:, 4*ii+1]
        z_rel = data[:, 2] - data[:, 4*ii+2]
        x_rel = x * np.cos(phi0) + y * np.sin(phi0)
        y_rel = x * -np.sin(phi0) + y * np.cos(phi0)

        dist = np.linalg.norm((x_rel, y_rel, z_rel), axis=0)
        ax3.plot(t, dist, color=color, linewidth=1.5)

    # Configure trajectory subplot
    ax1.set_aspect('equal')
    ax1.plot(arena[0]/2*np.cos(np.linspace(0, 2*np.pi, 500)),
              arena[0]/2*np.sin(np.linspace(0, 2*np.pi, 500)), color='gray', linewidth=0.5)
    ax1.set_xlim([-arena[0]/2-100, arena[0]/2+100])
    ax1.set_ylim([-arena[0]/2-100, arena[0]/2+100])
    ax1.axis('off')
    ax1.set_title('n_magnitude_leds = {}'.format(value), fontsize=13)

    # Configure distance subplot
    ax3.set(xlabel='Time (s)')
    ax3.set_ylim([0, 2000])
    ax3.axhline(target_distance, color='gray', linestyle='--', linewidth=1, zorder=0)

dist_row[0].set_ylabel('Distance (mm)', fontsize=12)

# Single legend for the whole figure (one entry per follower, colors shared across columns)
handles, labels = traj_row[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='outside upper center', ncol=len(handles),
                     frameon=False, title=formation, fontsize=14, title_fontsize=15)
for line in legend.get_lines():
    line.set_linewidth(3)

if not os.path.exists('./logfigs'):
    os.makedirs('./logfigs')

fig.savefig('./logfigs/sweep_n_magnitude_leds_large_comparison.png', dpi=300, bbox_inches='tight', transparent=True)
fig.savefig('./logfigs/sweep_n_magnitude_leds_large_comparison.svg', bbox_inches='tight', transparent=True)
print('Saved comparison plot to ./logfigs/sweep_n_magnitude_leds_large_comparison.{png,svg}')

plt.show()
