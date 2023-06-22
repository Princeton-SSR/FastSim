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
import ipyvolume as ipv
import matplotlib.cm as cm
import argparse

from environment import Environment
from dynamics import Dynamics
from lib_heap import Heap


def log_meta(filename):
    """Logs the meta data of the experiment
    """
    meta = {'Experiment': experiment_file, 'Number of fishes': no_fish, 'Simulation time [s]': simulation_time, 'Clock frequency [Hz]': clock_freq, 'Arena [mm]': arena_list, 'Visual range [mm]': v_range, 'Width of blindspot [mm]': w_blindspot, 'Radius of blocking sphere [mm]': r_sphere, 'Visual noise magnitude [% of distance]': n_magnitude}
    with open('./logfiles/{}_meta.txt'.format(filename), 'w') as f:
        json.dump(meta, f, indent=2)

def simulate(influence_param: float, sensing_angle: float):

    pos = np.zeros((no_fish, 4))
    vel = np.zeros((no_fish, 4))
    pos[:,:2] = initial_spread * (np.random.rand(no_fish, 2) - 0.5) + arena_center[:2] # x,y
    pos[:,2] = 10 * np.random.rand(1, no_fish) # z, all fish at same noise-free depth results in LJ lock
    pos[:,3] = 2*math.pi * (np.random.rand(1, no_fish) - 0.5) # phi
    
    fish_specs = (v_range, w_blindspot, r_sphere, n_magnitude)
    environment = Environment(pos, vel, fish_specs, arena)
    dynamics = Dynamics(environment)
    H = Heap(no_fish)

    # Create Fish Instances And Insert Into Heap
    fishes = []
    
    for fish_id in range(no_fish):
        clock = random.gauss(clock_rate, 0.1*clock_rate)
        fishes.append(Fish(fish_id, dynamics, environment, speed_up, attract, sensing_angle))
        H.insert(fish_id, clock)

    # Simulate
    print('#### WELCOME TO BLUESIM ####')
    print('Progress:', end=' ', flush=True)
    t_start = time.time()
    simulation_steps = no_fish*simulation_time*clock_freq # overall
    steps = 0
    prog_incr = 0.1

    print("Initial positions [x,y,z,theta]")
    print(pos)

    while True:
        progress = steps/simulation_steps
        if progress >= prog_incr:
            print('{}%'.format(round(prog_incr*100)), end=' ', flush=True)
            prog_incr += 0.1
        if steps >= simulation_steps:
                break

        (uuid, event_time) = H.delete_min()
        duration = random.gauss(clock_rate, 0.1*clock_rate)
        fishes[uuid].run(duration)
        H.insert(uuid, event_time + duration)

        steps += 1

    print('| Duration: {} sec\n -'.format(round(time.time()-t_start)))

    # Save Data
    # filename = time.strftime("%y%m%d_%H%M%S") # date_time
    filename = "influence_param_{}_sensing_param_{}_attract_{}_speed_up_{}_no_fish_{}_spread_{}_time_{}".format(influence_param, sensing_angle, attract, speed_up, no_fish, initial_spread, simulation_time)
    print("Filename")
    environment.log_to_file(filename)
    log_meta(filename=filename)

    print('Simulation data got saved in ./logfiles/{}_data.txt,\nand corresponding experimental info in ./logfiles/{}_meta.txt.\n -'.format(filename, filename))

    return filename

def animate(filename: str):

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
    #colors[:, 0, :] = cm.Reds(0.5) # this fish is red

    # Create Animation
    fig = ipv.figure()
    ipv.xlim(0, arena[0])
    ipv.ylim(0, arena[1])
    ipv.zlim(0, arena[2])
    ipv.style.use('dark')

    quiver = ipv.quiver(x, y, z, np.cos(phi), np.sin(phi), np.zeros((1,len(phi))),size=6, color=colors[:,:,:3])
    ipv.animation_control(quiver, interval=clock_rate)

    ipv.save('./animations/{}_animation.html'.format(filename))

    print('BLUEANIMAT saved your animation in ./animations/{}_animation.html.\nOpen with your favorite browser, sit back and enjoy the extravaganza!'.format(filename))

    return 

if __name__ == "__main__":

    speed_up = 1
    attract = 1

    # Experimental Parameters
    no_fish = 8
    simulation_time = 1750 # [s]
    clock_freq = 2 # [Hz]
    clock_rate = 1/clock_freq

    # Fish Specifications
    v_range=1000 # visual range, [mm]
    w_blindspot=50 # width of blindspot, [mm]

    r_sphere=50 # radius of blocking sphere for occlusion, [mm]
    n_magnitude=0.1 # visual noise magnitude, [% of distance]

    # Standard Tank
    # arena_list = [1780, 1780, 1170]
    #TODO: Add circle as arena shape (cylinder)
    arena_list = [45000,45000,500]
    arena = np.array(arena_list)
    arena_center = arena / 2.0

    # Standard Surface Initialization
    initial_spread = 2000

    parser = argparse.ArgumentParser()
    parser.add_argument("--speed_up", type=int)
    parser.add_argument("--attract", type=int)
    parser.add_argument("--no_fish", type=int)
    parser.add_argument("--spread", type=float)
    parser.add_argument("--simulation_time", type=int)
    parser.add_argument("--clock_freq", type=int)
    parser.add_argument("--v_range", type=int)
    parser.add_argument("--w_blindspot", type=int)
    parser.add_argument("--r_sphere", type=int)
    parser.add_argument("--n_magnitude", type=float)
    parser.add_argument("exp_filename", type=str) # you must pass this argument in 

    args = parser.parse_args()

    experiment_file = args.exp_filename

    #import Fish class directly from module specified by experiment type
    Fish = getattr(importlib.import_module('fishfood.' + experiment_file), 'Fish') 
    
    # Conditonally update relevant arguments based on passed in arguments

    if args.speed_up != None:
        speed_up = args.speed_up

    if args.attract != None:
        attract = args.attract

    if args.no_fish != None:
        no_fish = args.no_fish

    if args.spread != None:
        initial_spread = args.spread

    if args.simulation_time != None:
        simulation_time = args.simulation_time
    
    if args.v_range != None:
        v_range = args.v_range

    if args.w_blindspot != None:
        w_blindspot = args.w_blindspot

    if args.r_sphere != None:
        r_sphere = args.r_sphere
    
    if args.n_magnitude != None:
        n_magnitude = args.n_magnitude

    influence_param_arr = np.linspace(.2, 3.5, 8)
    # Failed Experiment 1: Sensing angle np.linspace(15, 90, 10) and all four behaviors didn't work well at all 
    sensing_angle_arr = np.linspace(15, 90, 8)
    
    for i in range(2):
        for j in range(2):
            attract = i
            speed_up = j

            for influence_param in influence_param_arr:
             for sensing_angle in sensing_angle_arr:
                filename = simulate(influence_param, sensing_angle)
                animate(filename=filename)