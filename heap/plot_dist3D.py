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

from sklearn.neighbors import NearestNeighbors

# filename = '240219_213513'
# Read meta file

# filename = '250129_094252'  #"F2S"
filename = []
# filename = '250129_121536'


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

no_leader = 1
## at time t1
# followers

def get_follower_data(data, ii):
    # ii is the robot id
    x = data[:, 4*ii]
    y = data[:, 4*ii+1]
    z = data[:, 4*ii+2]
    return x, y, z

def plot_neighbors( X,Y,Z,t):

    # plt.plot(target_points[0:t, 0], target_points[0:t, 1])
    # plt.plot(q_mt_x1_old, q_mt_y1_old, 'ro', color='green')
    # plt.plot(nodes[:, 0], nodes[:, 1], 'ro')
    N = X.shape[0]

    # Stack into Nx3 array of points
    points = np.stack((X, Y, Z), axis=1)

    # Find k nearest neighbors (excluding itself)
    k = 2  # number of neighbors to connect
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    # Draw lines to neighbors
    for i in range(N):
        for j in indices[i][1:]:  # skip the first one (itself)
            ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]], color='gray', linewidth=0.5)


# Create 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Select timestep to visualize
tt = int(timesteps/4)

# Plot followers
for ii in range(no_leader, fishes):
    x1, y1, z1 = get_follower_data(data, ii)
    ax.scatter(x1[tt], y1[tt], z1[tt], label=f'Follower {ii}', color='blue', s=100)

# Plot leader
x0, y0, z0 = get_follower_data(data, 0)
ax.scatter(x0[tt], y0[tt], z0[tt], label='Leader', color='red', s=100)

# plot_neighbors(x1, y1, z1, tt)

# Customize plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Set equal aspect ratio
ax.set_box_aspect([1,1,1])

plt.show()