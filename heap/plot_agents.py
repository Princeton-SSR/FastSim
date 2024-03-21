#!/usr/bin/python
"""Animates simulation data from logfiles with ipyvolume

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
no_leader = meta['Number of leaders']
t = np.arange(0, timesteps,1) /clock_freq
y_offset = 400 # to offset ax2

# figure setup
# Set the font to be recognizable
plt.rcParams['svg.fonttype'] = 'none'

fig1, ax1s = plt.subplots(1, 4,constrained_layout = True,figsize=(10,3))
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
gs = ax1s[2].get_gridspec()
ax1s[2].remove()
ax1s[3].remove()
ax1s[2] = fig1.add_subplot(gs[-2:])
# fig1.figsize=(10, 3)
fig2, ax2s = plt.subplots(1, 1)

# assign axis
ax1 = ax1s[0]
ax2 = ax1s[1]
ax3 = ax1s[2]
ax4 = ax2s

# Build your secondary mirror axes:
# fig2, (ax3, ax4) = plt.subplots(1, 2)
# map1 = ax3.imshow(np.stack([t, t]),cmap='Oranges')
# map2 = ax4.imshow(np.stack([t, t]),cmap='Blues')
# ax3.axis('off')
# ax4.axis('off')
# fig1.colorbar(map1,values=t,ax=ax1)
# fig1.colorbar(map2,values=t,ax=ax1)

for i_trial in range(0,no_trial):

    # Read Experimental Parameters
    # if no_trial == 1:
    #     data = np.loadtxt('./logfiles/{}_data.txt'.format(filename), delimiter=',')
    # else:
    data = np.loadtxt('./logfiles/{}_{}_data.txt'.format(filename,i_trial), delimiter=',')

    ################################################

    
    # followers
    for ii in range(no_leader,fishes):
        x = data[:, 4*ii]
        y = data[:, 4*ii+1]
        z = data[:, 4*ii+2]
        ax1.scatter(x, y, c=t, s=5, cmap='Oranges', alpha=0.8)  # Use color based on time

    for ii in range(0,no_leader):
        x = data[:, 4*ii]
        y = data[:, 4*ii+1]
        z = data[:, 4*ii+2]
        ax1.scatter(x, y, c=t, s=5, cmap='Blues', alpha=1)  # Use color based on time


    ############################################
    # Leader states
    x0 = data[:, 0]
    y0 = data[:, 1]
    z0 = data[:, 2]
    phi0 = data[:, 3]

    # plot leader
    ax2.arrow(-75,0-y_offset,150,0,linewidth = 1,head_width=80, fc='b',ec='b')

    # plot trajectories
    for ii in range(no_leader,fishes):
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
        ax2.scatter(x_rel, y_rel-y_offset, c=t, s=5, cmap='Oranges', alpha=0.8)  # Use color based on time

        # plot time series
        ax4.plot(t,x_rel,c='red')
        ax4.plot(t,y_rel,c='g')
        ax4.plot(t,z_rel,c='b')
        ax4.plot(t,phi_rel,c='black')

        # plot distance, don't plot more than 10 
        if i_trial < 10:
        # if i_trial > 10 and i_trial < 20:
            ax3.plot(t,np.linalg.norm((x_rel,y_rel,z_rel),axis=0))


# Customize grid, legend, etc. (optional)

# Customize grid, legend, etc. (optional)
# ax1.grid(True)
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

ax3.set_xlim([0, 500])
ax3.set_ylim([0, 2500])
# ax3.set_yticks([0, 1000, 2000])


ax4.legend(('x','y','z','$\phi$'),loc='upper right')
ax4.set(xlabel='t')
ax4.set_title('relative states')

# ax3.legend(('x','y','z'))
ax3.set(xlabel='t (s)')
ax3.set(ylabel='distance (mm)')
# ax3.grid('on')
# Save image
# fig1.savefig('animations/{}_trajectory.png'.format(filename),facecolor='white',transparent=False,dpi=500)  # Replace with desired filename


# remove axis to save vector file size
ax1.remove()
ax2.remove()
fig1.savefig('animations/{}_trajectory.svg'.format(filename),facecolor='white',transparent=False,dpi=500)  # vector file but it's huge

# ax3.set_xlim([150, 250])
# ax3.set_xticks([150, 250])

# ax3.set_ylim([50, 450])
# ax3.set_yticks([50, 450])

# fig1.savefig('animations/{}_trajectory_zoomed.svg'.format(filename),facecolor='white',transparent=False,dpi=500)  # vector file but it's huge


# Display plot (optional)
plt.show()
   