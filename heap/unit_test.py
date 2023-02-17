import numpy as np

from environment import *

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

# basic test of two robots to the right
def test_environment_count_left_right_1():
    # the number of fish
    n = 3
    # self is an environment
    env = set_up_environment(n)
    # set self position
    env.pos[0] = np.array([0,0,100,np.pi/2])
    # source_id is a number
    source_id = 0
    # robots is a list of indices (excluding self)
    robots = np.arange(1,n)
    # rel_pos is the relative positions as a list of [x,y] I think
    # let self be at [0,0,np.pi/2]
    rel_pos = np.array([[0,0,0,0],[100,100,100,0],[100,200,100,0]])

    left_count, right_count = env.count_left_right(source_id, robots, rel_pos)

    assert left_count == 0
    assert right_count == 2

# basic test of several robots around
def test_environment_count_left_right_2():
    # the number of fish
    n = 6
    # self is an environment
    env = set_up_environment(n)
    # set self position
    env.pos[0] = np.array([0,0,100,np.pi/2])
    # source_id is a number
    source_id = 0
    # robots is a list of indices (excluding self)
    robots = np.arange(1,n)
    # rel_pos is the relative positions as a list of [x,y] I think
    # let self be at [0,0,np.pi/2]
    rel_pos = np.array([[0,0,0,0],[100,100,100,0],[100,200,100,0],[-100,0,100,0],[-50,-200,100,0],[-1,100,100,0]])

    left_count, right_count = env.count_left_right(source_id, robots, rel_pos)

    assert left_count == 3
    assert right_count == 2

def test_environment_count_left_right_3():
    pass

def test_environment_count_left_right_4():
    pass

def test_environment_count_left_right_5():
    pass




######### main #################

print("starting count_left_right tests")
test_environment_count_left_right_1()
test_environment_count_left_right_2()
test_environment_count_left_right_3()
test_environment_count_left_right_4()
test_environment_count_left_right_5()
print("passed count_left_right_tests")
