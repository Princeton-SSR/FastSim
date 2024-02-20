"""Runs experiments. Change parameters here.

Attributes:
    arena (np-array of ints): Arena size in mm, [l, w, h]
    arena_center (np-array of floats): Arena center
    clock_freq (int): Fish update frequency
    clock_rate (float): Fish update rate
    dynamics (class instance): Fish dynamics
    environment (class instance): Fish environment
    filename (string): Time-prefix of logfiles, yymmdd_hhmmss
    fishes (list of class instances): Fishes
    H (class instance): Min heap
    initial_spread (int): Area spread of fish at initialization
    no_fish (int): Number of simulated fishes
    pos (np-array of floats): Fish positions at initialization, no_fish x [x,y,z,phi]
    prog_incr (float): Description
    simulation_steps (float): Required number of steps to simulate no_fish with clock_freq for simulation_time
    simulation_time (int): Experiment time in s
    steps (int): Simulation steps counter
    t_start (float): Experiment start time
    vel (np-array of floats): Fish velocities at initialization, no_fish x [vx,vy,vz,vphi]
"""
import json
import math
import numpy as np
import random
import sys
import time
import importlib

from environment import Environment
from dynamics import Dynamics
from lib_heap import Heap
import os

def log_meta(fn):
    """Logs the meta data of the experiment
    """
    meta = {'Experiment': experiment_file, 'Number of fishes': no_fish, 'Number of trials': no_trial, 'Simulation time [s]': simulation_time, 'Clock frequency [Hz]': clock_freq, 'Arena [mm]': arena_list, 'Visual range [mm]': v_range, 'Width of blindspot [mm]': w_blindspot, 'Radius of blocking sphere [mm]': r_sphere, 'Visual noise magnitude [% of distance]': n_magnitude}
    with open('./logfiles/{}_meta.txt'.format(fn), 'w') as f:
        json.dump(meta, f, indent=2)

# Read Experiment Description
try:
    experiment_file = sys.argv[1]
except:
    print('Please provide the filename of the experiment you want to simulate, e.g.:\n >python simulation.py dispersion')
    sys.exit()

#import Fish class directly from module specified by experiment type
Fish = getattr(importlib.import_module('fishfood.' + experiment_file), 'Fish') 

## Feel free to loop over multiple simulations with different parameters! ##

# Experimental Parameters
no_fish = 15
no_fish = getattr(importlib.import_module('fishfood.' + experiment_file), 'N_fish', no_fish)  # overwrite if the experiment file specify
simulation_time = 600 # [s]
clock_freq = 2 # [Hz]
clock_rate = 1/clock_freq # [s]
no_trial = 1 # number of simulations performed 
filename = time.strftime("%y%m%d_%H%M%S") # date_time

# Fish Specifications
v_range=5000 # visual range, [mm]
w_blindspot=50 # width of blindspot, [mm]
# w_blindspot=3141 # TODO: figure out mapping mm to degrees
r_sphere=50 # radius of blocking sphere for occlusion, [mm]
n_magnitude=0.1 # visual noise magnitude, [% of distance]
fish_specs = (v_range, w_blindspot, r_sphere, n_magnitude)

# Standard Tank
# arena_list = [1780, 1780, 1170]
#TODO: Add circle as arena shape (cylinder)
arena_list = [10000,10000,500]
arena = np.array(arena_list)
arena_center = arena / 2.0

# repeating trials
for i_trial in range(no_trial):

    # seed random generator
    random.seed(i_trial) # for heap
    np.random.seed(i_trial) # for initial condition

    # Standard Surface Initialization
    initial_spread = 5000
    pos = np.zeros((no_fish, 4))
    vel = np.zeros((no_fish, 4))
    pos[:,:2] = initial_spread * (np.random.rand(no_fish, 2) - 0.5) + arena_center[:2] # x,y
    pos[:,2] = 10 * np.random.rand(1, no_fish) # z, all fish at same noise-free depth results in LJ lock
    pos[:,3] = 2*math.pi * (np.random.rand(1, no_fish) - 0.5) # phi

    # fix leader pos
    pos[0,:2] = arena_center[:2]
    pos[0,2] = 0
    pos[0,3] = 0

    # Create Environment, Dynamics, And Heap
    environment = Environment(pos, vel, fish_specs, arena)
    dynamics = Dynamics(environment)
    H = Heap(no_fish)

    # Create Fish Instances And Insert Into Heap
    fishes = []
    for fish_id in range(no_fish):
        clock = random.gauss(clock_rate, 0.1*clock_rate)
        fishes.append(Fish(fish_id, dynamics, environment))
        H.insert(fish_id, clock)

    # Simulate
    print('#### WELCOME TO BLUESIM ####')
    print("Start trial "+str(i_trial+1))
    print('Progress:', end=' ', flush=True)
    t_start = time.time()
    simulation_steps = no_fish*simulation_time*clock_freq # overall
    steps = 0
    prog_incr = 0.1

    # print("Initial positions [x,y,z,theta]")
    # print(pos)

    # Main block for eular integration 
    # Note that the "step" here is not exactly the time step. For each time step, there are no_fish steps
    while True:

        # Displaying and keeping track of progress
        progress = steps/simulation_steps
        if progress >= prog_incr:
            print('{}%'.format(round(prog_incr*100)), end=' ', flush=True)
            prog_incr += 0.1
        if steps >= simulation_steps:
                break

        # time step for one fish 
        (uuid, event_time) = H.delete_min() # pull a fish from the heap
        duration = random.gauss(clock_rate, 0.1*clock_rate)
        fishes[uuid].run(duration) # RUN THE TIMESTEP FOR THE FISH
        H.insert(uuid, event_time + duration) # return the fish to the heap after updating its clock

        steps += 1

    print('| Duration: {} sec\n -'.format(round(time.time()-t_start)))

    # Save Data
    environment.log_to_file(filename+"_"+str(i_trial))
    log_meta(filename+"_"+str(i_trial))

    print('Simulation data got saved in ./logfiles/{}_data.txt,\nand corresponding experimental info in ./logfiles/{}_meta.txt.\n -'.format(filename, filename))
    # print('Create corresponding animation by running >python animation.py {}'.format(filename))
    # print('#### GOODBYE AND SEE YOU SOON AGAIN ####')

    # Run animation right after the code
    os.system(f'python animation.py '+filename+"_"+str(i_trial))

# Run agent plots right after the code
os.system(f'python plot_agents.py '+filename)