
import json
import numpy as np
import sys
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# Load Data
try:
    filename = sys.argv[1]
except:
    print('Provide prefix of data you want to animate in format yymmdd_hhmmss as command line argument, e.g.:\n >python animation.py 201005_111211')
    sys.exit()
try:
    data = np.loadtxt('./logfiles/{}_data.txt'.format(filename), delimiter=',')
    with open('./logfiles/{}_meta.txt'.format(filename), 'r') as f:
        meta = json.loads(f.read())
except:
    print('Data file with prefix {} does not exist.\nProvide prefix of data you want to animate in format yymmdd_hhmmss as command line argument, e.g.:\n >python animation.py 201005_111211'.format(filename))
    sys.exit()

# Read Experimental Parameters
clock_freq = meta['Clock frequency [Hz]']
clock_rate = 1000/clock_freq # [ms]
arena = meta['Arena [mm]']
timesteps = data.shape[0]
no_fish = int(data.shape[1]/8)
no_leader = meta['Number of leaders']

# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
# ax.axis('off')

# Setting the axes properties
ax.set_xlim3d([0, arena[0]])
ax.set_xlabel('X')
ax.set_ylim3d([0, arena[1]])
ax.set_ylabel('Y')
ax.set_zlim3d([0, arena[2]])
ax.set_zlabel('Z')

# choose a different color for each trajectory
colors = plt.cm.turbo(np.linspace(0, 1, no_fish))
colors[:no_leader,:] = colors[0,:] # make all the follower the same color (red)
colors[no_leader:,:] = colors[-1,:] # make all the follower the same color (red)

# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

line_shadows = sum([ax.plot([], [], [], '-', c=1-0.5*(1-c))
           for c in colors], [])
pt_shadows   = sum([ax.plot([], [], [], 'o', c=1-0.5*(1-c))
           for c in colors], [])

# title = ax.title('')
# line_shadows = sum([ax.plot([], [], [], '-', c=c, alpha=0.8)
#            for c in colors], [])
# pt_shadows = sum([ax.plot([], [], [], 'o', c=c, alpha=0.8)
#            for c in colors], [])

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.invert_zaxis()
ax.view_init(45, 15)

# animation function.  This will be called sequentially with the frame number
def animate(i):

    # to skip frames
    i = i*N_skip

    # plot shadows first
    for line, pt, ls, ps,  i_fish in zip(lines, pts, line_shadows, pt_shadows, range(no_fish)):
        # x, y, z = data[:i].T
        x = data[:i, 4*i_fish]
        y = data[:i, 4*i_fish+1]
        z = data[:i, 4*i_fish+2]

        ls.set_data(x, y)
        ls.set_3d_properties(arena[2])
    
        ps.set_data(x[-1:], y[-1:])
        ps.set_3d_properties(arena[2])

    for line, pt, ls, ps,  i_fish in zip(lines, pts, line_shadows, pt_shadows, range(no_fish)):
        # x, y, z = data[:i].T
        x = data[:i, 4*i_fish]
        y = data[:i, 4*i_fish+1]
        z = data[:i, 4*i_fish+2]


        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(45, 15+0.1 * i)
    fig.canvas.draw()
    return lines + pts + line_shadows + pt_shadows

# ax.title("t = {}".format(1))
# ax.title("t = ")

# set up animation
# Parameters for video
N_skip = 5 # only plotting every N_skip frames
N = np.int32(data.shape[0]/N_skip)
N = 100
print("Number of frames to be plotted = ",N)
fps = 10

ani = animation.FuncAnimation(fig, animate, N, interval=10, blit=True)
ani.save('./animations/{}_animation.mp4'.format(filename), fps=fps, extra_args=['-vcodec', 'libx264'])
# plt.show()