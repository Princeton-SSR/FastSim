#!/usr/bin/python
"""Animates simulation data from logfiles_LF with ipyvolume

Attributes:
    clock_freq (float): Clock frequency
    clock_rate (float): Clock rate
    colors (np-array of floats): Colors fish depending on their location
    fig1 (fig1ure object): ipv fig1ure
    fishes (int): Number of simulated fishes
    phi (float): Orientation angles
    quiver (plot object): ipv quiver plot
    timesteps (TYPE): Description
    v (float): Position magnitude
    x (float): x-positions
    y (float): y-positions
    z (float): z-positions
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

# Read meta file
filename = []
# filename = '250721_163249'

if not filename:
    try:
        filename = sys.argv[1]
    except:
        print('Provide prefix of data you want to animate in format yymmdd_hhmmss as command line argument, e.g.:\n >python plot_agents.py 240219_213513')
        sys.exit()

# Load Data

try:
    data = np.loadtxt('./logfiles_LF/{}_data.txt'.format(filename), delimiter=',')
    with open('./logfiles_LF/{}_meta.txt'.format(filename), 'r') as f:
        meta = json.loads(f.read())
except:
    print('Data file with prefix {} does not exist.\nProvide prefix of data you want to animate in format yymmdd_hhmmss as command line argument, e.g.:\n >python animation.py 201005_111211'.format(filename))
    sys.exit()


clock_freq = meta['Clock frequency [Hz]']
clock_rate = 1000/clock_freq # [ms]
arena = meta['Arena [mm]']
timesteps = data.shape[0]
fishes = int(data.shape[1]/8)
# no_leader = meta['Number of leaders']
t = np.arange(0, timesteps,1) /clock_freq
y_offset = 400 # to offset ax2


no_leader = 1
no_trial = 1


data = np.loadtxt('./logfiles_LF/{}_data.txt'.format(filename), delimiter=',')


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig1, ax1s = plt.subplots(2, 4,constrained_layout = True,figsize=(15,6))
# assign axis
ax1 = ax1s[0,0]
ax2 = ax1s[0,1]
ax3 = ax1s[0,2]
ax4 = ax1s[0,3]

cmaps = ['Greys', 'Purples', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu']

# for ii in range(0, fishes):
#     x = data[:, 4*ii]
#     y = data[:, 4*ii+1]
#     z = data[:, 4*ii+2]

#     if ii == 0:
#         cmaps.append('Blues')
#         ax.scatter(x, y, c=t, s=5, cmap='Blues', alpha=1)  # Use color based on time

#     else: 
#         if ii > len(cmaps) - 1:
#             cmaps.append(cmaps[ii % len(cmaps)])  # Cycle through colormaps if more fishes than colormaps
#         ax.scatter(x, y, c=t, s=1, cmap=cmaps[ii], alpha=0.8)  # Use color based on time





# followers
for ii in range(no_leader,fishes):
    x = data[:, 4*ii]
    y = data[:, 4*ii+1]
    z = data[:, 4*ii+2]
    # ax1.scatter(x, y, c=t, s=5, cmap='Oranges', alpha=0.8)  # Use color based on time

    if ii > len(cmaps) - 1:
        cmaps.append(cmaps[ii % len(cmaps)])  # Cycle through colormaps if more fishes than colormaps
        
    ax.scatter(x, y, c=t, s=1, cmap=cmaps[ii], alpha=0.8)  # Use color based on time
    ax1.scatter(x, y, c=t, s=1, cmap=cmaps[ii], alpha=0.8)  # Use color based on time


for ii in range(0,no_leader):
    x = data[:, 4*ii]
    y = data[:, 4*ii+1]
    z = data[:, 4*ii+2]
    ax.scatter(x, y, c=t, s=5, cmap='Blues', alpha=1)  # Use color based on time
    ax1.scatter(x, y, c=t, s=5, cmap='Blues', alpha=1)  # Use color based on time



############################################
# Leader states
x0 = data[:, 0]
y0 = data[:, 1]
z0 = data[:, 2]
phi0 = data[:, 3]


i_trial = 1

# figure setup
# Set the font to be recognizable
plt.rcParams['svg.fonttype'] = 'none'




# plot trajectories
for ii in range(no_leader,fishes):
    x = data[:, 4*ii]-x0
    y = data[:, 4*ii+1]-y0   
    z_rel = data[:, 4*ii+2] - z0    
    phi_rel = data[:, 4*ii+3] - phi0
    phi_rel[phi_rel < -2*np.pi] += 2*np.pi
    phi_rel[phi_rel >  2*np.pi] -= 2*np.pi

    x, y, z_rel = x0 - data[:, 4 * ii], y0 - data[:, 4 * ii + 1], z0 - data[:, 4 * ii + 2] 
    phi = data[:, 4 * ii + 3]
    x_rel = x * np.cos(phi) + y * np.sin(phi)
    y_rel = x * -np.sin(phi) + y * np.cos(phi)
    bearing = np.degrees(( np.arctan2( y_rel, x_rel,) + np.pi) % (2 * np.pi) - np.pi )
    pitch = np.arctan2(z_rel, np.sqrt(y_rel**2 + x_rel**2)) * 180 / np.pi
    # x_rel = x*np.cos(phi0)+y*np.sin(phi0)
    # y_rel = -x*np.sin(phi0)+y*np.cos(phi0)

    # plot relative position
    ax2.scatter( - x_rel, - y_rel-y_offset, c=t, s=5, cmap='Oranges', alpha=0.8)  # Use color based on time

    # # plot time series
    # ax4.plot(t,x_rel,c='red')
    # ax4.plot(t,y_rel,c='g')
    # ax4.plot(t,z_rel,c='b')
    # ax4.plot(t,phi_rel,c='black')

    ax1s[1,0].plot(t,x_rel, label=f'x_rel 0 - {ii}')
    ax1s[1,1].plot(t,y_rel, label=f'y_rel 0 - {ii}')
    ax1s[1,2].plot(t,z_rel, label=f'z_rel 0 - {ii}')

    ax1s[1,3].plot(t,bearing, label=f'Trial {i_trial}')
    ax4.plot(t,pitch, label=f'Trial {i_trial}')

    dist = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
    ax3.plot(t, dist, label=f'Follower {ii}')

# plot leader
ax2.arrow(-75,0-y_offset,130,np.pi/2,linewidth = 1,head_width=80, fc='b',ec='b')

# ax2.arrow(-75,0-y_offset,-130,np.pi/2,linewidth = 1,head_width=80, fc='black',ec='black')

ax2.scatter(0, -y_offset, c='blue', s=10, label='Leader', alpha=0.8)

# plot scale bar
ax2.plot([75, 335], [1000-y_offset, 1000-y_offset], color='r', linewidth=1)
ax2.text(0,1000-250-y_offset,'260 mm',horizontalalignment='left', 
         verticalalignment='center', fontsize=10)

set_limits = 300
ax2.set_xlim([-set_limits, set_limits])
ax2.set_ylim([-set_limits, set_limits])



# Customize grid, legend, etc. (optional)

# Customize grid, legend, etc. (optional)
# ax1.grid(True)
ax.set_aspect('equal')
ax.plot(arena[0]/2*np.cos(np.linspace(0, 2*np.pi, 500)),arena[0]/2*np.sin(np.linspace(0, 2*np.pi, 500)),color='gray')
# Hide only the rectangle frame
for side in ['top', 'right']: #, 'bottom', 'left']:
    ax.spines[side].set_visible(False)

# Set aspect ratio to be equal
ax1.set_aspect('equal')
# ax1.set_title('trajectory')
ax1.set(xlabel='$x_{global}$',ylabel='$y_{global}}$')

ax1.plot(arena[0]/2*np.cos(np.linspace(0, 2*np.pi, 500)),arena[0]/2*np.sin(np.linspace(0, 2*np.pi, 500)),color='gray')
ax1.set_xlim([-arena[0]/2-100, arena[0]/2+100])
ax1.set_ylim([-arena[0]/2-100, arena[0]/2+100])
ax1.axis('off')


# ax2.grid(True)
ax2.set_aspect('equal')
# ax2.set_title('relative trajectory')
ax2.set(xlabel='$x_{leader}  (mm)$',ylabel='$y_{leader}  (mm)$')
ax2.set_xlim([-1500, 1500])
ax2.set_ylim([-1500, 1500])
ax2.set_yticks([-1000, 0, 1000])
ax2.axis('off')

# ax3.set_xlim([0, 200])
# ax3.set_ylim([0, 2000])
# ax3.set_yticks([0, 1000, 2000])
# ax3.legend(('1','2','3','4','5','6'))
ax3.set(xlabel='t (s)')
ax3.set(ylabel='distance to leader (id = 0) (mm)')
# ax3.grid('on')
# Save image


ax1s[1,0].legend()
ax1s[1,0].set(xlabel='t (s)')
ax1s[1,0].set(ylabel='x_rel (mm)')
ax1s[1,1].legend()
ax1s[1,1].set(xlabel='t (s)')
ax1s[1,1].set(ylabel='y_rel (mm)')
ax1s[1,2].legend()
ax1s[1,2].set(xlabel='t (s)')
ax1s[1,2].set(ylabel='z_rel (mm)')
ax1s[1,3].legend()
ax1s[1,3].set(xlabel='t (s)')
ax1s[1,3].set(ylabel='bearing (degree)')

ax4.set(xlabel='t (s)')
ax4.set(ylabel='pitch (degree)')


from itertools import combinations

def plot_inter_agent_distances(data, num_agents=3, show_plot=True, use_2d=False):
    """
    Plots inter-agent distances over time for all pair combinations.

    Parameters:
        data (np.ndarray): Array of shape (T, 4 * num_agents) containing [x, y, z, phi] for each agent.
        num_agents (int): Number of agents in the data.
        show_plot (bool): Whether to display the plot.
        use_2d (bool): If True, compute distance using only (x, y); if False, use (x, y, z).
    
    Returns:
        dict: Dictionary with keys as agent pairs and values as distance arrays over time.
    """
    time_steps = data.shape[0]
    dim = 2 if use_2d else 3
    positions = np.zeros((num_agents, time_steps, dim))

    for ii in range(num_agents):
        positions[ii, :, 0] = data[:, 4 * ii]
        positions[ii, :, 1] = data[:, 4 * ii + 1]
        if not use_2d:
            positions[ii, :, 2] = data[:, 4 * ii + 2]

    # Compute pairwise distances
    pairwise_distances = {}
    agent_pairs = list(combinations(range(num_agents), 2))
    for (i, j) in agent_pairs:
        dist = np.linalg.norm(positions[i] - positions[j], axis=1)
        pairwise_distances[(i, j)] = dist

    # Plot
    if show_plot:
        plt.figure(figsize=(10, 6))
        for (i, j), dist in pairwise_distances.items():
            plt.plot(dist, label=f'Distance: Agent {i} - Agent {j}')
        plt.xlabel('Time step')
        plt.ylabel('Distance')
        plt.title('Inter-Agent Distances Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return pairwise_distances

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def plot_nearest_neighbor_distances(data, num_agents=3, show_plot=True, use_2d=False):
    """
    For each agent, plots the distance to its nearest neighbor over time.

    Parameters:
        data (np.ndarray): Array of shape (T, 4 * num_agents) with [x, y, z, phi] for each agent.
        num_agents (int): Number of agents.
        show_plot (bool): Whether to display the plot.
        use_2d (bool): If True, compute distance using only (x, y); otherwise use (x, y, z).

    Returns:
        np.ndarray: Array of shape (num_agents, T) with nearest neighbor distances for each agent over time.
    """
    time_steps = data.shape[0] 
    dim = 2 if use_2d else 3
    positions = np.zeros((num_agents, time_steps, dim))

    for ii in range(num_agents):
        positions[ii, :, 0] = data[:, 4 * ii]
        positions[ii, :, 1] = data[:, 4 * ii + 1]
        if not use_2d:
            positions[ii, :, 2] = data[:, 4 * ii + 2]

    # Compute nearest neighbor distances
    nn_distances = np.zeros((num_agents, time_steps))  # (agent, time)

    for tt in range(time_steps):
        for i in range(num_agents):
            # Compute distance from agent i to all other agents at time t
            dists = [
                np.linalg.norm(positions[i, tt] - positions[j, tt])
                for j in range(num_agents) if j != i
            ]
            nn_distances[i, tt] = min(dists)



    # Plot
    if show_plot:
        plt.figure(figsize=(10, 6))
        for i in range(num_agents):
            plt.plot( t , nn_distances[i], label=f'Agent {i} NN Distance')
       
       
        # Add horizontal line at 260 mm
        body_length = 130  # Assuming body length is 130 mm
        d = 1 * body_length

        for i in range(num_agents):
            plt.axhline(y= d , color='r', linestyle='--', label='260 mm')
            plt.axhline(y= d *np.sqrt(2), color='r', linestyle='--', label='260 sq(2) mm')
            break # only plot once

        plt.ylim(-100, 1000)  # Set y-axis limits from 0 to 1000
        plt.xlabel('Time step')
        plt.ylabel('Distance to Nearest Neighbor')
        plt.title('Each Agent\'s Nearest Neighbor Distance Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()

    return nn_distances

def plot_phi_data():
    test_filename = "/Users/di/Documents/GitHub/FastSim/heap/fishfood/flocking_helper/debug_test/phi_data.csv"
    phi_data = pd.read_csv(test_filename, names=['id', 'phi_a_value'])
    phi_data = phi_data.iloc[1:, :]

    unique_ids = np.unique(phi_data.iloc[:, 0])
    num_unique_ids = len(unique_ids)    

    figure, ax = plt.subplots(figsize=(10, 6))

    for id in unique_ids:
        mask = phi_data['id'] == id
        values = phi_data[mask]['phi_a_value'].astype(float)

        # Ensure we plot the minimum length between t and values to avoid index errors
        min_length = min(len(t[1:]), len(values))
        ax.scatter(t[1:min_length+1], values[:min_length], label=f'ID {id}')

    ax.set_xlabel('time_steps')
    ax.set_ylabel('phi_a_value')
    ax.set_title('phi_a_value for each ID')
    ax.legend()


for ii in range(0,fishes):
    x = data[:, 4*ii]
    y = data[:, 4*ii+1]
    z = data[:, 4*ii+2]
    phi = data[:, 4*ii + 3]
    
    vx = data[:, 4*ii + 4]
    vy = data[:, 4*ii + 5]
    vz = data[:, 4*ii + 6]
    vphi = data[:, 4*ii + 7]
    v = np.sqrt(vx**2 + vy**2 + vz**2)

def plot_speed(data, num_agents, t):
    """Plots position and velocity components of each agent over time in a 2x4 subplot."""
    fig1, axs = plt.subplots(3, 4, figsize=(15, 6), constrained_layout=True)

    for ii in range(num_agents):
        # Extract position data
        x = data[:, 4*ii]
        y = data[:, 4*ii+1]
        z = data[:, 4*ii+2]
        phi = data[:, 4*ii+3]
        
        # Extract velocity data
        vx = data[:, 4*num_agents + ii + 1]
        vy = data[:, 4*num_agents + ii + 2]
        vz = data[:, 4*num_agents + ii + 3]
        # vphi = data[:, 4*ii + 7] # this is not used
        v = np.sqrt(vx**2 + vy**2 + vz**2)

        # Plotting
        axs[0, 0].plot(t, x, label=f'Agent {ii}')
        axs[0, 1].plot(t, y, label=f'Agent {ii}')
        axs[0, 2].plot(t, z, label=f'Agent {ii}')
        axs[0, 3].plot(t, phi, label=f'Agent {ii}')

        axs[1, 0].plot(t, vx, label=f'Agent {ii}')
        axs[1, 1].plot(t, vy, label=f'Agent {ii}')
        axs[1, 2].plot(t, vz, label=f'Agent {ii}')
        axs[1, 3].plot(t, v, label=f'Agent {ii}')  # Speed
        # axs[1, 3].plot(t, vphi, label=f'Agent {ii}') # velocity of phi is not available in data

        axs[2, 0].plot(t, vx/130, label=f'Agent {ii}')
        axs[2, 1].plot(t, vy/130, label=f'Agent {ii}')
        axs[2, 2].plot(t, vz/130, label=f'Agent {ii}')
        axs[2, 3].plot(t, v/130, label=f'Agent {ii}')  # Speed
        # axs[1, 3].plot(t, vphi, label=f'Agent {ii}') # velocity of phi is not available in data

    # Titles and labels
    axs[0, 0].set_title('X Position')
    axs[0, 1].set_title('Y Position')
    axs[0, 2].set_title('Z Position')
    axs[0, 3].set_title('Orientation (phi)')

    axs[1, 0].set_title('Velocity in X')
    axs[1, 1].set_title('Velocity in Y')
    axs[1, 2].set_title('Velocity in Z')
    axs[1, 3].set_title('Speed (V)')

    ylabels = ['Position (mm)', 'Velocity (mm/s)', 'Velocity (BL/s)']
    for i, ax_row in enumerate(axs):
        for ax in ax_row:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabels[i])
            # ax.legend()
            ax.grid(True)
    axs[0, 0].legend(loc='upper right', fontsize='small')

    # fig1.tight_layout() # use constrained_layout instead
    # plt.show() # remove show here, so it can plot with other plots

# plot_speed(data, fishes, t)

# plot_inter_agent_distances(data, num_agents=fishes, show_plot=True, use_2d=False)
plot_nearest_neighbor_distances(data, num_agents=fishes, show_plot=True, use_2d=False)

# plot_phi_data()

# plot_speed(data, fishes, t)

plt.show()
   