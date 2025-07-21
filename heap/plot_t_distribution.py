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



for ii in range(0,no_leader):
    x0 = data[:, 4*ii]
    y0 = data[:, 4*ii+1]
    z0 = data[:, 4*ii+2]

plt.figure(figsize=(10, 10))
tt = int( timesteps/4 )

# x1, y1, z1 = get_follower_data(data, 1)
# plt.scatter(x1[tt], y1[tt], label='follower 1', color='blue', s=100)

for ii in range(no_leader,fishes):
    x1, y1, z1 = get_follower_data(data, ii)
    plt.scatter(x1[tt], y1[tt], label='Leader {}'.format(ii), color='blue', s=100)

plt.scatter(x0[tt], y0[tt], label='Leader', color='red', s=100)
plt.legend()
# plt.plot(x[tt], y[tt], label='Follower {}'.format(ii), color='blue')

plt.show()