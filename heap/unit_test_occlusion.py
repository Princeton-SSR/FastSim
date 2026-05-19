
import math
import random
import numpy as np
from scipy.spatial.distance import cdist
import sys
import pdb
import os

i_trial = 0
random.seed(i_trial+1) # for heap
np.random.seed(i_trial) # for initial condition

w_blindspot = 50 # width of blindspot, [mm]
no_fish = 10
no_robots = no_fish
no_states = 4
initial_spread = 2000 # radius

leader_initial = [00, -2000, 0, math.pi/2]
leader_initial = [1000, -2800, 0, math.pi * 0/4]

pos = np.zeros((no_fish, 4))
theta = np.random.rand(no_fish) * math.pi + math.pi* 2/2
r = np.random.rand(no_fish) * initial_spread 

pos[:,0] = - r * np.cos(theta) + leader_initial[0]
pos[:,1] = r * np.sin(theta) + leader_initial[1]
# pos[:,2] = 10 * np.random.rand(1, no_fish) # z, all fish at same noise-free depth results in LJ lock
pos[:,3] = 2*math.pi * (np.random.rand(1, no_fish) - 0.5) # phi

pos[0,0] = leader_initial[0] #arena_center[:2] 
pos[0,1] = leader_initial[1]
pos[0,2] = leader_initial[2]
pos[0,3] = leader_initial[3]




def blind_spot(source_id, robots, rel_pos):
    """Omits fishes within the blind spot behind own body
    """
    r_blockage = w_blindspot/2

    phi = pos[source_id,3]
    phi_xy = [math.cos(phi), math.sin(phi)]
    # mag_phi = np.linalg.norm(phi_xy)
    mag_phi = 1 # Ko: this value is always one, don't know why it's calcualted
    # print(mag_phi)
    
    candidates = robots.copy()
    for robot in candidates:

        print(" ------ in enviornment/blind_spot, print robot and rel_pos ------")
        print("source robot:", source_id, "robot:", robot)
        print("rel_pos:", rel_pos[robot,:2]) 
        print("dot", np.dot(phi_xy, rel_pos[robot,:2])) 


        print("dot", np.dot(phi_xy, pos[robot,:2])) 

        print("dot", np.dot([1,0], rel_pos[robot,:2]))  
        print("################") 

        dot = np.dot(phi_xy, rel_pos[robot,:2])
        if dot < 0:
            d_robot = np.linalg.norm(rel_pos[robot,:2])

            angle = abs(math.acos(dot / (mag_phi * d_robot))) - math.pi / 2 # cos(a-b) = ca*cb+sa*sb = sa

            if  math.cos(angle) * d_robot < r_blockage:
                robots.remove(robot)

def rel_pos(source_id, robots, pos):
    """Calculate relative position of other fishes to self
    """
    # Relative positions
    pos_others = np.reshape(pos, (1,no_robots*no_states))
    pos_self = np.tile(pos[source_id,:], (1,no_robots))
    rel_pos = pos_others - pos_self
    # rel_pos[source_id,:] = rel_pos # row
    # rel_pos_ = np.reshape(rel_pos, (no_robots, no_states))
    # rel_pos[:,source_id*no_states:source_id*no_states+no_states] = -rel_pos_ # columns
    
source_id = 3
robots = set(range(no_fish)) # all robots
# robots = {0} # only the leaders have leds
robots.discard(source_id) # discard self


import matplotlib.pyplot as plt

plt.quiver(pos[:,0], pos[:,1], np.cos(pos[:,3]), np.sin(pos[:,3]))

# Scatter plot with different colors for each agent
colors = plt.cm.jet(np.linspace(0, 1, no_fish))
plt.scatter(pos[:,0], pos[:,1], c=colors, label='Agents')

# Add robot ID labels
for i in range(no_fish):
    plt.text(pos[i,0], pos[i,1], str(i), fontsize=12, color='black')

plt.scatter(pos[0,0], pos[0,1], c='y', label='Leader', edgecolors='black', s=100) # Highlight leader
plt.axis('equal')
plt.legend()
plt.show()
