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

# some corner cases, still same starting position
def test_environment_count_left_right_3():
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
    rel_pos = np.array([[0,0,100,np.pi/2],[0,100,100,0],[0,-200,100,0],[1,-200,100,0],[0,1100,100,0],[0,900,500,0]])
    # [0,0,1,0,0]

    left_count, right_count = env.count_left_right(source_id, robots, rel_pos)
    # print(right_count)

    assert left_count == 0
    assert right_count == 1

# basic test first quadrant
def test_environment_count_left_right_4():
    # the number of fish
    n = 6
    # self is an environment
    env = set_up_environment(n)
    # set self position
    env.pos[0] = np.array([0,0,100,np.pi/4])
    # source_id is a number
    source_id = 0
    # robots is a list of indices (excluding self)
    robots = np.arange(1,n)
    # rel_pos is the relative positions as a list of [x,y] I think
    # let self be at [0,0,np.pi/2]
    rel_pos = np.array([[0,0,100,np.pi/4],[100,100,100,0],[100,200,100,0],[-100,0,100,0],[-50,-200,100,0],[-1,100,100,0]])

    left_count, right_count = env.count_left_right(source_id, robots, rel_pos)

    assert left_count == 3
    assert right_count == 1

# basic test second quadrant
def test_environment_count_left_right_5():
    # the number of fish
    n = 6
    # self is an environment
    env = set_up_environment(n)
    # set self position
    env.pos[0] = np.array([0,0,100,3*np.pi/4])
    # source_id is a number
    source_id = 0
    # robots is a list of indices (excluding self)
    robots = np.arange(1,n)
    # rel_pos is the relative positions as a list of [x,y] I think
    # let self be at [0,0,np.pi/2]
    rel_pos = np.array([[0,0,100,3*np.pi/4],[100,100,100,0],[100,200,100,0],[-100,0,100,0],[-50,-200,100,0],[-1,100,100,0]])

    left_count, right_count = env.count_left_right(source_id, robots, rel_pos)

    assert left_count == 2
    assert right_count == 3

# basic test third quadrant
def test_environment_count_left_right_6():
    # the number of fish
    n = 6
    # self is an environment
    env = set_up_environment(n)
    # set self position
    env.pos[0] = np.array([0,0,100,5*np.pi/4])
    # source_id is a number
    source_id = 0
    # robots is a list of indices (excluding self)
    robots = np.arange(1,n)
    # rel_pos is the relative positions as a list of [x,y] I think
    # let self be at [0,0,np.pi/2]
    rel_pos = np.array([[0,0,100,5*np.pi/4],[100,100,100,0],[100,200,100,0],[-100,0,100,0],[-50,-200,100,0],[-1,100,100,0]])

    left_count, right_count = env.count_left_right(source_id, robots, rel_pos)

    assert left_count == 1
    assert right_count == 3

    # basic test fourth quadrant
def test_environment_count_left_right_7():
    # the number of fish
    n = 6
    # self is an environment
    env = set_up_environment(n)
    # set self position
    env.pos[0] = np.array([0,0,100,7*np.pi/4])
    # source_id is a number
    source_id = 0
    # robots is a list of indices (excluding self)
    robots = np.arange(1,n)
    # rel_pos is the relative positions as a list of [x,y] I think
    # let self be at [0,0,np.pi/2]
    rel_pos = np.array([[0,0,100,7*np.pi/4],[100,100,100,0],[100,200,100,0],[-100,0,100,0],[-50,-200,100,0],[-1,100,100,0]])
    
    
    left_count, right_count = env.count_left_right(source_id, robots, rel_pos)

    assert left_count == 3
    assert right_count == 2

def test_environment_count_left_right_8():
    # random samples left and right should work 
    # num fish 
    n = 10
    env = set_up_environment(n)
    z = 100
    theta = np.pi / 2
    y = 0
    env.pos[0] = np.array([0,0,z,theta])
    source_id = 0

    robots = np.arange(1,n)
    rel_pos = np.zeros((n-1, 4))
    rel_pos[:,1] = y 
    rel_pos[:,2] = z
    rel_pos[:,3] = theta 
    rel_pos = np.insert(rel_pos, 0, env.pos[0], axis=0)
    for i in range(100):
        rel_pos[1:, 0] = np.random.uniform(-50, 50, n-1)
        # print(rel_pos[:,0])
        

        acc_left = np.count_nonzero(rel_pos[:, 0] < 0)
        acc_right = np.count_nonzero(rel_pos[:, 0] > 0)
        
        left_count, right_count = env.count_left_right(source_id, robots, rel_pos)
        # if(acc_left != left_count or acc_right != right_count):
        #     print(str(acc_left) + ", " + str(left_count) + "; " + str(acc_right) + ", " + str(right_count))
        assert left_count == acc_left
        assert right_count == acc_right 

# This isn't designed for specific edge cases, but should work very generally 
def test_environment_count_left_right_9():

    def rotate_about_angle(vec, angle_in_radians):
        R = np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                     [np.sin(angle_in_radians),  np.cos(angle_in_radians)]])
        
        return R @ vec

    n = 20
    env = set_up_environment(n)
    z = 100
    samp_theta = np.random.uniform(-2*np.pi, 2*np.pi)
    x_y_vec = np.random.uniform(-50, 50, 2)
    env.pos[0][:2] = x_y_vec
    env.pos[0][2] = z
    env.pos[0][3] = samp_theta
    source_id = 0

    robots = np.arange(1,n)
    rel_pos = np.zeros((n-1, 4))
    rel_pos = np.insert(rel_pos, 0, env.pos[0], axis=0)
    #for i in range(100):
    rel_pos[1:, :2] = np.random.uniform(-50, 50, (n-1, 2))
    rel_pos[1:, 2] = z
    rel_pos[1:, 3] = np.random.uniform(-2*np.pi, 2*np.pi)
    # center everything
    rel_pos[:, :2] -= x_y_vec

    # angle we're gonna rotate by 
    angle_shift = np.pi/2  - samp_theta 

    # this is essentially makes the 0th vector (source) the center 
    for i in range(n):
        rel_pos[i, :2] = rotate_about_angle(rel_pos[i, :2], angle_shift)
    
    print(rel_pos)
    rel_pos[0, 3] += angle_shift
    acc_left = np.count_nonzero(rel_pos[:, 0] < 0)
    acc_right = np.count_nonzero(rel_pos[:, 0] > 0)

    # undo all of these

    for i in range(n):
        rel_pos[i, :2] = rotate_about_angle(rel_pos[i, :2], -angle_shift)

    rel_pos[:, :2] += x_y_vec
    rel_pos[0, 3] -= angle_shift
    left_count, right_count = env.count_left_right(source_id, robots, rel_pos)

    assert left_count == acc_left
    assert right_count == acc_right 

######### main #################

print("starting count_left_right tests")
test_environment_count_left_right_1()
test_environment_count_left_right_2()
test_environment_count_left_right_3()
test_environment_count_left_right_4()
test_environment_count_left_right_5()
test_environment_count_left_right_6()
test_environment_count_left_right_7()
test_environment_count_left_right_8()
test_environment_count_left_right_9()
print("passed count_left_right_tests")
