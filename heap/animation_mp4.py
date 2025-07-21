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
import ipyvolume as ipv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import sys


# Load Data
try:
    filename = sys.argv[1]
except:
    print('Provide prefix of data you want to animate in format yymmdd_hhmmss as command line argument, e.g.:\n >python animation.py 201005_111211')
    sys.exit()
try:
    data = np.loadtxt('./logfiles_LF/{}_data.txt'.format(filename), delimiter=',')
    with open('./logfiles_LF/{}_meta.txt'.format(filename), 'r') as f:
        meta = json.loads(f.read())
except:
    print('Data file with prefix {} does not exist.\nProvide prefix of data you want to animate in format yymmdd_hhmmss as command line argument, e.g.:\n >python animation.py 201005_111211'.format(filename))
    sys.exit()

# Read Experimental Parameters
clock_freq = meta['Clock frequency [Hz]']
clock_rate = 1000/clock_freq # [ms]
arena = meta['Arena [mm]']
timesteps = data.shape[0]
fishes = int(data.shape[1]/8)

# Format Data
x = data[:, :1]
y = data[:, 1:2]
z = data[:, 2:3]
phi = data[:, 3:4]
#vx = data[:, 4:5]
#vy = data[:, 5:6]
#vz = data[:, 6:7]

for ii in range(1,fishes):
    x = np.concatenate((x, data[:, 4*ii:4*ii+1]), axis=1)
    y = np.concatenate((y, data[:, 4*ii+1:4*ii+2]), axis=1)
    z = np.concatenate((z, data[:, 4*ii+2:4*ii+3]), axis=1)
    phi = np.concatenate((phi, data[:, 4*ii+3:4*ii+4]), axis=1)
    #vx = np.concatenate((vx, data[:, 4*(fishes+ii):4*(fishes+ii)+1]), axis=1)
    #vy = np.concatenate((vy, data[:, 4*(fishes+ii)+1:4*(fishes+ii)+2]), axis=1)
    #vz = np.concatenate((vz, data[:, 4*(fishes+ii)+2:4*(fishes+ii)+3]), axis=1)

# Colors
v = np.sqrt(x**2 + y**2 + z**2)
v -= v.min(); v /= v.max()
colors = np.array([cm.Blues(k) for k in v])

# colors[:, 0, :] = cm.Reds(0.5) # this fish is red
colors[:, 0, :] = cm.Wistia(0.3) # this fish is yellowish

# Create Animation
fig = ipv.figure()
ipv.xlim(-arena[0], arena[0])
ipv.ylim(-arena[1], arena[1])
ipv.zlim(0, arena[2])
ipv.style.use('dark')

quiver = ipv.quiver(x, y, z, np.cos(phi), np.sin(phi), np.zeros((1,len(phi))),size=6, color=colors[:,:,:3])
ipv.animation_control(quiver, interval=clock_rate)

ipv.save('./animations_LF/{}_animation.html'.format(filename))

# Note: ipv.movie() requires a Jupyter environment and doesn't work from command line
# Uncomment the next line if running from a Jupyter notebook:
# ipv.movie('./animations_LF/{}_animation.mp4'.format(filename), fps=30)

# Create MP4 animation using matplotlib
print('Creating MP4 animation...')
fig_mp4 = plt.figure(figsize=(10, 8))
ax = fig_mp4.add_subplot(111, projection='3d')

# Set axis limits
ax.set_xlim(-arena[0], arena[0])
ax.set_ylim(-arena[1], arena[1])
ax.set_zlim(0, arena[2])
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')

# Animation function
def animate_frame(frame):
    ax.clear()
    ax.set_xlim(-arena[0], arena[0])
    ax.set_ylim(-arena[1], arena[1])
    ax.set_zlim(0, arena[2])
    ax.set_xlabel('X [mm]', color='white')
    ax.set_ylabel('Y [mm]', color='white')
    ax.set_zlabel('Z [mm]', color='white')
    ax.set_facecolor('black')
    
    # Make axis text visible on black background
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    
    # Set view angle (elevation, azimuth)
    # You can change these values to adjust the camera angle
    ax.view_init(elev=60, azim= 20)  # elev: up/down angle, azim: rotation angle
    
    # Plot fish positions and orientations
    for fish in range(fishes):
        # Fish position
        fish_x = x[frame, fish]
        fish_y = y[frame, fish]
        fish_z = z[frame, fish]
        
        # Fish orientation
        fish_phi = phi[frame, fish]
        dx = np.cos(fish_phi) * 130  # Scale arrow length
        dy = np.sin(fish_phi) * 130
        dz = 80  # Fixed height for the arrow
        
        # Color based on fish index
        if fish == 0:
            color = 'yellow'
        else:
            color = 'blue'
        
        # Plot fish as a quiver arrow
        ax.quiver(fish_x, fish_y, fish_z, dx, dy, dz, 
                 color=color, arrow_length_ratio=0.3, linewidth=2)
        
        # Plot trajectory trail (last 20 points)
        start_idx = max(0, frame-20)
        trail_x = x[start_idx:frame+1, fish]
        trail_y = y[start_idx:frame+1, fish]
        trail_z = z[start_idx:frame+1, fish]
        ax.plot(trail_x, trail_y, trail_z, color=color, alpha=0.3, linewidth=1)
    
    ax.set_title(f'Fish Animation - Frame {frame}/{timesteps-1}', color='white')

# Create animation
fps = 30
interval = clock_rate if clock_rate > 10 else 50  # Minimum 50ms interval
anim = animation.FuncAnimation(fig_mp4, animate_frame, frames=timesteps, 
                              interval=interval, blit=False, repeat=False)

# Save as MP4
print(f'Saving MP4 animation to ./animations_LF/{filename}_animation.mp4...')
Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='FastSim'), bitrate=1800)
anim.save('./animations_LF/{}_animation.mp4'.format(filename), writer=writer)
plt.close(fig_mp4)

print('BLUEANIMAT saved your animation in ./animations_LF/{}_animation.html.\nAlso saved MP4 version in ./animations_LF/{}_animation.mp4.\nOpen with your favorite browser/player, sit back and enjoy the extravaganza!'.format(filename, filename))