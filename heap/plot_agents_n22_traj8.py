#!/usr/bin/python
"""Plots simulation data from logfiles

Attributes:
    clock_freq (float): Clock frequency
    clock_rate (float): Clock rate in milliseconds
    arena (list): Arena dimensions in mm
    fishes (int): Number of simulated fishes
    no_leader (int): Number of leaders
    timesteps (int): Number of timesteps
    phi (float): Orientation angles
    x, y, z (float): Position coordinates
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sys
import os

# # Get filename from command line argument
# try:
#     filename = sys.argv[1]
# except:
#     print('Provide prefix of data you want to plot in format yymmdd_hhmmss as command line argument, e.g.:\n >python plot_agents_clean.py 240219_213513')
#     sys.exit()

# filename = '260317_170443' ## n = 8 octagon
# filename = '260318_105206' ## n = 15
filename = '260806_162608' ## n = 22

# Read data
try:
    data = np.loadtxt('./logfiles/{}_data.txt'.format(filename), delimiter=',')
    with open('./logfiles/{}_meta.txt'.format(filename), 'r') as f:
        meta = json.loads(f.read())
    no_trial = 1
except:
    data = np.loadtxt('./logfiles/{}_0_data.txt'.format(filename), delimiter=',')
    with open('./logfiles/{}_0_meta.txt'.format(filename), 'r') as f:
        meta = json.loads(f.read())
    no_trial = meta['Number of trials']

# Extract metadata
clock_freq = meta['Clock frequency [Hz]']
clock_rate = 1000/clock_freq  # [ms]
arena = meta['Arena [mm]']
timesteps = data.shape[0]
fishes = int(data.shape[1]/8)
no_leader = meta['Number of leaders']
t = np.arange(0, timesteps, 1) / clock_freq
y_offset = 000  # Offset for relative trajectory plot

# Figure setup
plt.rcParams['svg.fonttype'] = 'none'  # Set font to be recognizable in SVG

# Create main figure with multiple subplots
fig1, axes = plt.subplots(1, 5, constrained_layout=True, figsize=(15, 3))

# Assign axes
ax1 = axes[0]  # Global trajectory
ax2 = axes[1]  # Relative trajectory
ax3 = axes[2]  # Distance over time
ax4 = axes[3]  # Bearing
ax5 = axes[4]  # Pitch

for i_trial in range(no_trial):
    # Read trial data
    data = np.loadtxt('./logfiles/{}_{}_data.txt'.format(filename, i_trial), delimiter=',')


    
    # Plot followers
    for ii in range(no_leader, fishes):
        x = data[:, 4*ii]
        y = data[:, 4*ii+1]
        z = data[:, 4*ii+2]
        ax1.scatter(x, y, c=t, s=5, cmap='Blues', alpha=0.8)

    # Plot leaders
    for ii in range(no_leader):
        x = data[:, 4*ii]
        y = data[:, 4*ii+1]
        z = data[:, 4*ii+2]
        ax1.scatter(x, y, c=t, s=5, cmap='Oranges', alpha=1)

    # Extract leader states
    x0 = data[:, 0]
    y0 = data[:, 1]
    z0 = data[:, 2]
    phi0 = data[:, 3]

    # Find inflection points of the leader's figure-8 (where path curvature changes
    # sign, i.e. where the leader crosses through the center of the "8"). Smooth
    # the path first so fin-actuation jitter doesn't create spurious sign flips.
    window = min(31, timesteps - 1 + (timesteps % 2))
    if window % 2 == 0:
        window -= 1
    window = max(window, 5)
    x0_smooth = savgol_filter(x0, window, 3)
    y0_smooth = savgol_filter(y0, window, 3)

    vx0 = np.gradient(x0_smooth, t)
    vy0 = np.gradient(y0_smooth, t)
    ax0 = np.gradient(vx0, t)
    ay0 = np.gradient(vy0, t)

    speed_sq = vx0**2 + vy0**2
    curvature = np.divide(vx0*ay0 - vy0*ax0, speed_sq**1.5,
                           out=np.zeros_like(speed_sq), where=speed_sq > 1e-6)

    # The figure-8 only truly changes curvature sign where it crosses through its own
    # center; away from there, bang-bang fin actuation makes the *realized* curvature
    # jitter in sign too (e.g. overshoot at the outer tips of each lobe), which is not
    # a real inflection of the "8" shape. Gate sign-change detection to a core region
    # around the path's centroid so only the true crossings are picked up.
    center_xy = np.array([np.mean(x0), np.mean(y0)])
    dist_center = np.hypot(x0 - center_xy[0], y0 - center_xy[1])
    near_center = dist_center < 0.25 * np.max(dist_center)

    sign_change = np.where((np.diff(np.sign(curvature)) != 0) & near_center[:-1] & near_center[1:])[0]

    # Each crossing still produces a short burst of sign flips rather than one clean
    # flip. Chain-merge sign changes within `min_gap` of their neighbor into a single
    # group, then report the group's median index as that crossing's inflection point.
    min_gap = int(10.0 * clock_freq)  # samples; 10 s, well under the ~half-period between crossings
    groups = []
    for idx in sign_change:
        if groups and (idx - groups[-1][-1]) <= min_gap:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    inflection_idx = [int(np.median(g)) for g in groups]
    inflection_t = t[inflection_idx]

    # Draw leader fish on relative trajectory plot
    body_length = 130  # mm
    ax2.arrow(+0.5, 0-y_offset/body_length, -1, 0, linewidth=1, head_width=80/body_length, fc='orange', ec='orange')

    # Plot relative trajectories for each follower
    for ii in range(no_leader, fishes):
        # Calculate relative positions from follower's perspective
        x = x0 - data[:, 4*ii]
        y = y0 - data[:, 4*ii+1]
        z_rel = z0 - data[:, 4*ii+2]
        phi = data[:, 4*ii+3]
        phi_rel = data[:, 4*ii+3] - phi0
        
        # Normalize angles to [-2π, 2π]
        phi_rel[phi_rel < -2*np.pi] += 2*np.pi
        phi_rel[phi_rel > 2*np.pi] -= 2*np.pi

        # Transform to follower's reference frame
        x_rel = x * np.cos(phi) + y * np.sin(phi)
        y_rel = x * -np.sin(phi) + y * np.cos(phi)
        
        # Calculate bearing and pitch
        bearing = np.degrees((np.arctan2(y_rel, x_rel) + np.pi) % (2 * np.pi) - np.pi)
        pitch = np.arctan2(z_rel, np.sqrt(y_rel**2 + x_rel**2)) * 180 / np.pi

        # Plot relative position trajectory
        ax2.scatter(x_rel/body_length, (y_rel-y_offset)/body_length, c=t, s=5, cmap='Blues', alpha=0.8)
        # ax2.text(x_rel[0]/body_length, (y_rel[0]-y_offset)/body_length, str(ii), fontsize=12, color='black')
        ax2.set_aspect('equal')

        # # Draw final orientation arrow
        # dx = np.cos(phi_rel[-1])
        # dy = np.sin(phi_rel[-1])
        # ax2.arrow(x_rel[-1]/body_length, (y_rel[-1]-y_offset)/body_length, -dx, -dy, head_width=30/body_length, head_length=20/body_length, fc='orange', ec='orange')

        # Plot bearing and pitch over time
        ax4.plot(t, bearing, label=f'Follower {ii}')
        ax5.plot(t, pitch, label=f' {ii}')

        # Plot distance over time (limit to first 10 trials for clarity)
        if i_trial < 10:
            ax3.plot(t, np.linalg.norm((x_rel, y_rel, z_rel), axis=0), label=f' {ii}')

    # Add robot ID labels to global trajectory plot
    for ii in range(no_leader, fishes):
        ax1.text(data[0, 4*ii], data[0, 4*ii+1], str(ii), fontsize=12, color='black')

# Configure plot 1: Global trajectory
ax1.set_aspect('equal')
ax1.set(xlabel='$x_{global}$', ylabel='$y_{global}$')
ax1.plot(arena[0]/2*np.cos(np.linspace(0, 2*np.pi, 500)), 
         arena[0]/2*np.sin(np.linspace(0, 2*np.pi, 500)), color='gray')
ax1.set_xlim([-arena[0]/2-100, arena[0]/2+100])
ax1.set_ylim([-arena[0]/2-100, arena[0]/2+100])
ax1.axis('off')

# Configure plot 2: Relative trajectory
ax2.set_aspect('equal')
ax2.set(xlabel='$x$ (BL)', ylabel='$y$ (BL)')

angles = [60, -60, 90, -90, 120, -120, 180]
polar_length = 200/body_length
ax2.scatter(0,0, c='orange', s=20, zorder=5)  # Leader position
for angle in angles:
    x_point = polar_length * np.cos(np.radians(angle))
    y_point = polar_length * np.sin(np.radians(angle))
    ax2.scatter(-x_point, y_point - y_offset, c='red', s=5, zorder=5)

# Configure plot 3: Distance over time
ax3.set(xlabel='Time (s)')
# ax3.set_ylim([0, 2000])
# ax3.legend(ncol=3, fontsize='small', handlelength=1)

# Mark leader trajectory-8 inflection points (curvature sign flips)
for k, it in enumerate(inflection_t):
    ax3.axvline(it, color='k', linestyle='--', alpha=0.5, linewidth=1,
                label='Leader inflection' if k == 0 else None)

# Configure plot 4: Bearing
# ax4.legend()
ax4.set(xlabel='Time (s)')

# Configure plot 5: Pitch
ax5.legend(ncol=3, fontsize='small', handlelength=1)
ax5.set(xlabel='Time (s)')
# Display plots
plt.show()
   

# Check if logfigs folder exists, create if not
if not os.path.exists('./logfigs'):
    os.makedirs('./logfigs')

fig1.savefig('./logfigs/{}_plot.png'.format(filename), dpi=300, bbox_inches='tight', transparent=True)
fig1.savefig('./logfigs/{}_plot.svg'.format(filename), bbox_inches='tight', transparent=True)