#!/usr/bin/python
"""Plots simulation data from a sweep of n_magnitude_pos values for comparison.

Reads the manifest written by sweep_n_magnitude_pos.py (mapping each swept
n_magnitude_pos value to the logfile prefix it produced) and overlays all
runs in one figure, colored from light (low noise) to dark (high noise).

Usage:
    python plot_sweep_n_magnitude_pos.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from fishfood.exp_formations import CONFIGS

MANIFEST_PATH = './logfiles/sweep_n_magnitude_pos_manifest.json'
FORMATION_CONFIG = os.environ.get('FORMATION_CONFIG', 'F1S')  # sweep_n_magnitude_pos.py doesn't vary formation; matches its default

# Sequential blue ramp, light->dark, one step per swept value (see dataviz skill).
SEQUENTIAL_STEPS = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']

with open(MANIFEST_PATH, 'r') as f:
    manifest = json.load(f)

# Sort by noise magnitude so color always runs light (low noise) -> dark (high noise)
sweep_values = sorted(manifest.keys(), key=float)
colors = SEQUENTIAL_STEPS if len(sweep_values) <= len(SEQUENTIAL_STEPS) else \
    plt.cm.Blues(np.linspace(0.35, 0.95, len(sweep_values)))

plt.rcParams['svg.fonttype'] = 'none'  # Set font to be recognizable in SVG

fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 3.5))
ax1, ax3 = axes  # global traj, distance

arena = None

for value, color in zip(sweep_values, colors):
    filename = manifest[value]
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

# Configure plot 1: Global trajectory
ax1.set_aspect('equal')
ax1.set(xlabel='$x_{global}$', ylabel='$y_{global}$')
ax1.plot(arena[0]/2*np.cos(np.linspace(0, 2*np.pi, 500)),
          arena[0]/2*np.sin(np.linspace(0, 2*np.pi, 500)), color='gray', linewidth=0.5)
ax1.set_xlim([-arena[0]/2-100, arena[0]/2+100])
ax1.set_ylim([-arena[0]/2-100, arena[0]/2+100])
ax1.axis('off')
ax1.set_title('Global trajectory', fontsize=12)

# Target reference line, from this formation's config (safe_distance, approach_distance, distance, angle, pitch_range)
_, _, target_distance, _, _ = CONFIGS[FORMATION_CONFIG]

# Configure plot 3: Distance over time
ax3.set(xlabel='Time (s)')
ax3.set_title('Distance (mm)', fontsize=12)
ax3.set_ylim([0, 2000])
ax3.axhline(target_distance, color='gray', linestyle='--', linewidth=1, zorder=0)

# Single legend for the whole figure (one entry per swept n_magnitude_pos value)
handles, labels = ax3.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=len(sweep_values),
                     frameon=False, title='n_magnitude_pos', fontsize=14, title_fontsize=15)
for line in legend.get_lines():
    line.set_linewidth(3)

if not os.path.exists('./logfigs'):
    os.makedirs('./logfigs')

fig.savefig('./logfigs/sweep_n_magnitude_pos_comparison.png', dpi=300, bbox_inches='tight', transparent=True)
fig.savefig('./logfigs/sweep_n_magnitude_pos_comparison.svg', bbox_inches='tight', transparent=True)
print('Saved comparison plot to ./logfigs/sweep_n_magnitude_pos_comparison.{png,svg}')

plt.show()
