import numpy as np

from environment import *
import fishfood.bv_align as bv
# from environment import Environment
from dynamics import Dynamics

######### test count_right ##############

# breaking out the setup functions because its a lot
# can probably reduce some of this and parameterize parts to differentiate trials

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

def test_bv_align_move_1():
    #NOTE: Pectoral left moves you to the left, pectoral right moves you to the right 
    # initialize trivial set of agent rel_pos
    # run a single iteration of move
    # there is some randomness to the step (I think) so we will assert that the agent moved in generally the correct direction, maybe with some bounds
    # can copy this test for more complicated positions or corner cases

    # PARAMETERS
    # num fish
    n=2
    # duration
    d = 1
    # source_id is a number
    source_id = 0

    # set up sim
    test_fish, env, dyn = set_up_sim(n, source_id)

    # set self position
    env.pos = np.array([[0,0,100,np.pi/2],[100,100,100,np.pi/2]])
    # robots is a list of indices (excluding self)
    robots = np.arange(1,n)
    # rel_pos is the relative positions as a list of [x,y,z,theta]
    # let self be at [0,0,np.pi/2]
    rel_pos = np.array([[0,0,0,0],[100,100,0,0]])
    
    # robots, rel_pos, dist, leds = test_fish.environment.get_robots(source_id=0)

    target_pos, vel = test_fish.move(robots, rel_pos, dist=0, duration=d, attract=1, speed_up=1)

    #print(env.pos[0])
    #print(target_pos)

    # assert the fish has turned to the right (from pi/2) and moved some amount to the right (positive x and y)
    assert target_pos[0] > 0
    assert target_pos[1] > 0
    assert target_pos[3] < np.pi/2

def test_bv_align_move_2():
    #NOTE: Pectoral left moves you to the left, pectoral right moves you to the right 
    # initialize trivial set of agent rel_pos
    # run a single iteration of move
    # there is some randomness to the step (I think) so we will assert that the agent moved in generally the correct direction, maybe with some bounds
    # can copy this test for more complicated positions or corner cases

    # PARAMETERS
    # num fish
    n=2
    # duration
    d = 1
    # source_id is a number
    source_id = 0

    # set up sim
    test_fish, env, dyn = set_up_sim(n, source_id)

    # set self position
    env.pos = np.array([[0,0,100,np.pi/2],[100,100,100,np.pi/2]])
    # robots is a list of indices (excluding self)
    robots = np.arange(1,n)
    # rel_pos is the relative positions as a list of [x,y,z,theta]
    # let self be at [0,0,np.pi/2]
    rel_pos = np.array([[0,0,0,0],[100,100,0,0]])
    
    target_pos, vel = test_fish.move(robots, rel_pos, dist=0, duration=d, attract=0, speed_up=1)

    #print(env.pos[0])
   # print(target_pos)

    # assert the fish has turned to the left (from pi/2) and moved some amount to the left and up (negative x and positive y)
    assert target_pos[0] < 0
    assert target_pos[1] > 0
    assert target_pos[3] > np.pi/2

def test_bv_align_move_3():
    #NOTE: Pectoral left moves you to the left, pectoral right moves you to the right 
    # initialize trivial set of agent rel_pos
    # run a single iteration of move
    # there is some randomness to the step (I think) so we will assert that the agent moved in generally the correct direction, maybe with some bounds
    # can copy this test for more complicated positions or corner cases

    # PARAMETERS
    # num fish
    n=2
    # duration
    d = 1
    # source_id is a number
    source_id = 0

    # set up sim
    test_fish, env, dyn = set_up_sim(n, source_id)

    # set self position
    env.pos = np.array([[0,0,100,np.pi/2],[100,100,100,np.pi/2]])
    # robots is a list of indices (excluding self)
    robots = np.arange(1,n)
    # rel_pos is the relative positions as a list of [x,y,z,theta]
    # let self be at [0,0,np.pi/2]
    rel_pos = np.array([[0,0,0,0],[100,100,0,0]])
    
    target_pos, vel = test_fish.move(robots, rel_pos, dist=0, duration=d, attract=0, speed_up=0)

    print(env.pos[0])
    print(target_pos)

    # assert the fish has turned to the left (from pi/2) and moved some amount to the left and up (negative x and positive y)
    assert target_pos[0] < 0
    assert target_pos[1] > 0
    assert target_pos[3] > np.pi/2

#TODO: Additional test cases for the sake of completeness

######### main #################

print("starting bv_align move tests")
test_bv_align_move_1()
test_bv_align_move_2()
test_bv_align_move_3()
print("passed bv_align move tests")
