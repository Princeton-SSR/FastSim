#!/usr/bin/python
"""Animates simulation data from logfiles with ipyvolume

Attributes:
    clock_freq (float): Clock frequency
    clock_rate (float): Clock rate
    colors (np-array of floats): Colors fish depending on their location
    fig (figure object): ipv figure
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
try:
    filename = sys.argv[1]
except:
    print('Provide prefix of data you want to animate in format yymmdd_hhmmss as command line argument, e.g.:\n >python plot_agents.py 240219_213513')
    sys.exit()
# read data
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
    # print('Data file with prefix {} does not exist.\nProvide prefix of data you want to animate in format yymmdd_hhmmss as command line argument, e.g.:\n >python animation.py 201005_111211'.format(filename))
    # sys.exit()

clock_freq = meta['Clock frequency [Hz]']
clock_rate = 1000/clock_freq # [ms]
arena = meta['Arena [mm]']
timesteps = data.shape[0]
fishes = int(data.shape[1]/8)
t = np.arange(0, timesteps,1) /clock_freq

# figure setup
fig, axs = plt.subplots(2, 2,constrained_layout = True)
fig.figsize=(10, 6)

for i_trial in range(0,no_trial):

    # Read Experimental Parameters
    # if no_trial == 1:
    #     data = np.loadtxt('./logfiles/{}_data.txt'.format(filename), delimiter=',')
    # else:
    data = np.loadtxt('./logfiles/{}_{}_data.txt'.format(filename,i_trial), delimiter=',')

    ################################################
    # plot trajectories
    for ii in range(0,fishes):
        x = data[:, 4*ii]
        y = data[:, 4*ii+1]
        z = data[:, 4*ii+2]
        axs[0,0].scatter(x, y, c=t, s=5)  # Use color based on time

    # Customize grid, legend, etc. (optional)
    # axs[0,0].grid(True)
    # Set aspect ratio to be equal
    axs[0,0].set_aspect('equal')
    axs[0,0].set_title('trajectory')
    axs[0,0].set(xlabel='$x_{global}$',ylabel='$y_{global}}$')


    ############################################
    # Leader states
    x0 = data[:, 0]
    y0 = data[:, 1]
    z0 = data[:, 2]
    phi0 = data[:, 3]

    # plot leader
    axs[1,0].arrow(-150,0,300,0,linewidth = 3,head_width=100, fc='b',ec='b')

    # plot trajectories
    for ii in range(1,fishes):
        x = data[:, 4*ii]-x0
        y = data[:, 4*ii+1]-y0   
        z_rel = data[:, 4*ii+2] - z0    
        phi_rel = data[:, 4*ii+3] - phi0
        phi_rel[phi_rel < -2*np.pi] += 2*np.pi
        phi_rel[phi_rel >  2*np.pi] -= 2*np.pi

        # if phi_rel < -2*np.pi:
        #     phi_rel += 2*np.pi
        # elif phi_rel > 2*np.pi:
        #     phi_rel -= 2*np.pi
        x_rel = x*np.cos(phi0)+y*np.sin(phi0)
        y_rel = -x*np.sin(phi0)+y*np.cos(phi0)

        # plot relative position
        axs[1,0].scatter(x_rel, y_rel, c=t, s=5)  # Use color based on time

        # plot time series
        axs[0,1].plot(t,x_rel,c='red')
        axs[0,1].plot(t,y_rel,c='g')
        axs[0,1].plot(t,z_rel,c='b')
        axs[0,1].plot(t,phi_rel,c='black')

        # plot distance
        axs[1,1].plot(t,np.linalg.norm((x_rel,y_rel,z_rel),axis=0))


# Customize grid, legend, etc. (optional)
# axs[1,0].grid(True)
axs[1,0].set_aspect('equal')
axs[1,0].set_title('relative trajectory')
axs[1,0].set(xlabel='$x_{leader}$',ylabel='$y_{leader}$')


axs[0,1].legend(('x','y','z','$\phi$'),loc='upper right')
axs[0,1].set(xlabel='t')
axs[0,1].set_title('relative states')

# axs[1,1].legend(('x','y','z'))
axs[1,1].set(xlabel='t')
axs[1,1].set_title('distance')

# Save image
fig.savefig('animations/{}_trajectory.png'.format(filename),facecolor='white',transparent=False,dpi=500)  # Replace with desired filename
# fig.savefig('animations/{}_trajectory.svg'.format(filename),facecolor='white',transparent=False,dpi=500)  # vector file but it's huge

# Display plot (optional)
plt.show()
   