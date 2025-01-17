import json
import numpy as np
import matplotlib.pyplot as plt
import sys

# # Check if filenames are provided as arguments
# if len(sys.argv) < 2:
#     print('Provide prefixes of data you want to animate in format yymmdd_hhmmss as command line arguments, e.g.:')
#     print('python plot_agents.py 240219_213513 240220_101010')
#     sys.exit()

# filenames = sys.argv[1:]

filenames = ['250117_184648', '250117_184720', '250117_184925']

# figure setup
plt.rcParams['svg.fonttype'] = 'none'
fig1, ax1s = plt.subplots(2, 4, constrained_layout=True, figsize=(10, 6))

# assign axes
ax1, ax2, ax3, ax4 = ax1s[0, 0], ax1s[0, 1], ax1s[0, 2], ax1s[0, 3]

y_offset = 400  # to offset ax2

for filename_idx, filename in enumerate(filenames):
    try:
        data = np.loadtxt(f'./logfiles/{filename}_data.txt', delimiter=',')
        with open(f'./logfiles/{filename}_meta.txt', 'r') as f:
            meta = json.loads(f.read())
        no_trial = 1
    except:
        data = np.loadtxt(f'./logfiles/{filename}_0_data.txt', delimiter=',')
        with open(f'./logfiles/{filename}_0_meta.txt', 'r') as f:
            meta = json.loads(f.read())
        no_trial = meta['Number of trials']

    clock_freq = meta['Clock frequency [Hz]']
    arena = meta['Arena [mm]']
    timesteps = data.shape[0]
    fishes = int(data.shape[1] / 8)
    no_leader = meta['Number of leaders']
    t = np.arange(0, timesteps, 1) / clock_freq

    color = plt.cm.jet(filename_idx / len(filenames))

    for i_trial in range(0, no_trial):
        if no_trial > 1:
            data = np.loadtxt(f'./logfiles/{filename}_{i_trial}_data.txt', delimiter=',')

        # Plot followers
        for ii in range(no_leader, fishes):
            x, y = data[:, 4*ii], data[:, 4*ii+1]
            ax1.scatter(x, y, c=t, s=5, cmap='Oranges', alpha=0.8)

        # Plot leaders
        for ii in range(0, no_leader):
            x, y = data[:, 4*ii], data[:, 4*ii+1]
            ax1.scatter(x, y, c=t, s=5, cmap='Blues', alpha=1)

        # Leader states
        x0, y0, z0, phi0 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        # Plot leader arrow
        ax2.arrow(-75, 0-y_offset, 150, 0, linewidth=1, head_width=80, fc='b', ec='b')

        # Plot relative trajectories
        for ii in range(no_leader, fishes):
            x, y, z_rel = x0 - data[:, 4*ii], y0 - data[:, 4*ii+1], z0 - data[:, 4*ii+2]
            phi = data[:, 4*ii+3]
            x_rel = x * np.cos(phi) + y * np.sin(phi)
            y_rel = x * -np.sin(phi) + y * np.cos(phi)
            bearing = np.degrees((np.arctan2(y_rel, x_rel) + np.pi) % (2 * np.pi) - np.pi)
            pitch = np.arctan2(z_rel, np.sqrt(y_rel**2 + x_rel**2)) * 180 / np.pi

            ax2.scatter(x_rel, y_rel-y_offset, c=t, s=5, cmap='Oranges', alpha=0.8)

            ax1s[1, 0].plot(t, x_rel, color=color, label=f'{filename} Trial {i_trial}')
            ax1s[1, 1].plot(t, y_rel, color=color, label=f'{filename} Trial {i_trial}')
            ax1s[1, 2].plot(t, z_rel, color=color, label=f'{filename} Trial {i_trial}')
            ax1s[1, 3].plot(t, bearing, color=color, label=f'{filename} Trial {i_trial}')
            ax4.plot(t, pitch, color=color, label=f'{filename} Trial {i_trial}')

            if i_trial < 10:
                ax3.plot(t, np.linalg.norm((x_rel, y_rel, z_rel), axis=0), color=color)

# Set up axes
ax1.set_aspect('equal')
ax1.set(xlabel='$x_{global}$', ylabel='$y_{global}$')
ax1.plot(arena[0]/2*np.cos(np.linspace(0, 2*np.pi, 500)), arena[0]/2*np.sin(np.linspace(0, 2*np.pi, 500)), color='gray')
ax1.set_xlim([-arena[0]/2-100, arena[0]/2+100])
ax1.set_ylim([-arena[0]/2-100, arena[0]/2+100])
ax1.axis('off')

ax2.set_aspect('equal')
ax2.set(xlabel='$x_{leader} (mm)$', ylabel='$y_{leader} (mm)$')
ax2.set_xlim([-1500, 1500])
ax2.set_ylim([-1500, 1500])
ax2.set_yticks([-1000, 0, 1000])
ax2.axis('off')

ax3.set_xlim([0, 200])
ax3.set_ylim([0, 2000])
ax3.set(xlabel='t (s)', ylabel='distance (mm)')

for i in range(2):
    for j in range(4):
        ax1s[i, j].legend()
        ax1s[i, j].set(xlabel='t (s)')

ax1s[1, 0].set(ylabel='x_rel (mm)')
ax1s[1, 1].set(ylabel='y_rel (mm)')
ax1s[1, 2].set(ylabel='z_rel (mm)')
ax1s[1, 3].set(ylabel='bearing (degree)')
ax4.set(ylabel='pitch (degree)')

# Save the figure
fig1.savefig('animations/multi_trajectory.png', facecolor='white', transparent=False, dpi=500)

plt.show()
