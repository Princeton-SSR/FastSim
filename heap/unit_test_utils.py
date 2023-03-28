import numpy as np

from environment import *
import fishfood.bv_align as bv
# from environment import Environment
from dynamics import Dynamics

def set_up_environment(n):
    # Experimental Parameters
    no_fish = n
    # simulation_time = 120 # [s]
    # clock_freq = 2 # [Hz]
    # clock_rate = 1/clock_freq

    # Fish Specifications
    v_range=1000 # visual range, [mm]
    w_blindspot=50 # width of blindspot, [mm]
    r_sphere=50 # radius of blocking sphere for occlusion, [mm]
    n_magnitude=0.1 # visual noise magnitude, [% of distance]
    fish_specs = (v_range, w_blindspot, r_sphere, n_magnitude)

    # Standard Tank
    # arena_list = [1780, 1780, 1170]
    arena_list = [5000,5000,500]
    arena = np.array(arena_list)
    arena_center = arena / 2.0

    # Standard Surface Initialization
    initial_spread = 2000
    pos = np.zeros((no_fish, 4))
    vel = np.zeros((no_fish, 4))
    pos[:,:2] = initial_spread * (np.random.rand(no_fish, 2) - 0.5) + arena_center[:2] # x,y
    pos[:,2] = 10 * np.random.rand(1, no_fish) # z, all fish at same noise-free depth results in LJ lock
    pos[:,3] = 2*math.pi * (np.random.rand(1, no_fish) - 0.5) # phi

    # Create Environment, Dynamics, And Heap
    environment = Environment(pos, vel, fish_specs, arena)

    return environment


def set_up_sim(n, fid):
    no_fish = n
    # simulation_time = 120 # [s]
    # clock_freq = 2 # [Hz]
    # clock_rate = 1/clock_freq

    # Fish Specifications
    v_range=1000 # visual range, [mm]
    w_blindspot=50 # width of blindspot, [mm]
    r_sphere=50 # radius of blocking sphere for occlusion, [mm]
    n_magnitude=0.1 # visual noise magnitude, [% of distance]
    fish_specs = (v_range, w_blindspot, r_sphere, n_magnitude)

    # Standard Tank
    # arena_list = [1780, 1780, 1170]
    arena_list = [5000,5000,500]
    arena = np.array(arena_list)
    arena_center = arena / 2.0

    # Standard Surface Initialization
    initial_spread = 2000
    pos = np.zeros((no_fish, 4))
    vel = np.zeros((no_fish, 4))
    # vel[:,2] = -10 # changing this changes the z coord
    # pos[:,:2] = initial_spread * (np.random.rand(no_fish, 2) - 0.5) + arena_center[:2] # x,y
    # pos[:,2] = 10 * np.random.rand(1, no_fish) # z, all fish at same noise-free depth results in LJ lock
    # pos[:,3] = 2*math.pi * (np.random.rand(1, no_fish) - 0.5) # phi

    # init env and dynamics
    environment = Environment(pos, vel, fish_specs, arena)
    dynamics = Dynamics(environment)
    # initialize a fish
    test_fish = bv.Fish(my_id=fid, dynamics=dynamics, environment=environment)

    return test_fish, environment, dynamics