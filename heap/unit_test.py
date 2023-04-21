import numpy as np
import unittest

from environment import *
import fishfood.bv_align as bv
# from environment import Environment
from dynamics import Dynamics
from unit_test_utils import *

# @unittest.skip("test")
class TestEnvironmentMethods(unittest.TestCase):

    # basic test of two robots to the right
    def test_environment_count_left_right_1(self):
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

        self.assertEqual(left_count,0)
        self.assertEqual(right_count,2)

    # basic test of several robots around
    def test_environment_count_left_right_2(self):
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

        # rel_pos[0] can be anything since we initilaize the valid candidates (fish whose position we actually check) via robots
        rel_pos = np.array([[0,0,0,0],[100,100,100,0],[100,200,100,0],[-100,0,100,0],[-50,-200,100,0],[-1,100,100,0]])

        left_count, right_count = env.count_left_right(source_id, robots, rel_pos)

        self.assertEqual(left_count,3)
        self.assertEqual(right_count,2)

    # some corner cases, still same starting position
    def test_environment_count_left_right_3(self):
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

        self.assertEqual(left_count,0)
        self.assertEqual(right_count,1)

    # basic test first quadrant
    def test_environment_count_left_right_4(self):
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

        self.assertEqual(left_count,3)
        self.assertEqual(right_count,1)

    # basic test second quadrant
    def test_environment_count_left_right_5(self):
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

        self.assertEqual(left_count,2)
        self.assertEqual(right_count,3)

    # basic test third quadrant
    def test_environment_count_left_right_6(self):
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

        self.assertEqual(left_count,1)
        self.assertEqual(right_count,3)

        # basic test fourth quadrant
    def test_environment_count_left_right_7(self):
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

        self.assertEqual(left_count,3)
        self.assertEqual(right_count,2)

    def test_environment_count_left_right_8(self):
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
            self.assertEqual(left_count,acc_left)
        self.assertEqual(right_count,acc_right)

    # This isn't designed for specific edge cases, but should work very generally 
    @unittest.skip("test_count_left_right_9 doesn't work right now, @Osa should fix this")
    def test_environment_count_left_right_9(self):

        def rotate_about_angle(vec, angle_in_radians):
            R = np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                        [np.sin(angle_in_radians),  np.cos(angle_in_radians)]])
            
            return R @ vec

        n = 50
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
        
        rel_pos[0, 3] += angle_shift
        acc_left = np.count_nonzero(rel_pos[:, 0] < 0)
        acc_right = np.count_nonzero(rel_pos[:, 0] > 0)

        # undo all of these

        for i in range(n):
            rel_pos[i, :2] = rotate_about_angle(rel_pos[i, :2], -angle_shift)

        rel_pos[:, :2] += x_y_vec
        rel_pos[0, 3] -= angle_shift
        left_count, right_count = env.count_left_right(source_id, robots, rel_pos)

        if(acc_left != left_count or acc_right != right_count):
            print(str(acc_left) + ", " + str(left_count) + "; " + str(acc_right) + ", " + str(right_count))

        self.assertEqual(left_count,acc_left)
        self.assertEqual(right_count,acc_right)

    def test_environment_angle_threshold_1(self):
        # """ Test that the environment/angle_threshold works correctly """
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
        # rel_pos is the relative positions as a list of [x,y,z,theta]
        # let self be at [0,0,100,np.pi/2]
        rel_pos = np.array([[0,0,0,0],[100,101,100,0],[100,200,100,0]])

        new_robots = env.angle_threshold(source_id, robots, rel_pos, 90)

        self.assertEqual(new_robots, [1,2])

    def test_environment_angle_threshold_2(self):
        # """ Test that the environment/angle_threshold works correctly """
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
        # rel_pos is the relative positions as a list of [x,y,z,theta]
        # let self be at [0,0,100,np.pi/2]
        rel_pos = np.array([[0,0,0,0],[100,0,100,0],[-100,200,100,0]])

        new_robots = env.angle_threshold(source_id, robots, rel_pos, 90)

        self.assertEqual(new_robots, [2])


class TestBVMethods(unittest.TestCase):

    def test_bv_align_move_1(self):
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
        test_fish, env, dyn = set_up_sim(n, source_id, attract=1, speed_up=1, angle=90)

        # set self position
        env.pos = np.array([[0,0,100,np.pi/2],[100,100,100,np.pi/2]])
        # robots is a list of indices (excluding self)
        robots = np.arange(1,n)
        # rel_pos is the relative positions as a list of [x,y,z,theta]
        # let self be at [0,0,np.pi/2]
        rel_pos = np.array([[0,0,0,0],[100,100,0,0]])
        
        # robots, rel_pos, dist, leds = test_fish.environment.get_robots(source_id=0)

        target_pos, vel = test_fish.move(robots, rel_pos, dist=0, duration=d, attract=1, speed_up=1)

        print(env.pos[0])
        print(target_pos)

        # assert the fish has turned to the right (from pi/2) and moved some amount to the right (positive x and y)
        self.assertTrue(target_pos[0] > 0)
        self.assertTrue(target_pos[1] > 0)
        self.assertTrue(target_pos[3] < np.pi/2)

    def test_bv_align_move_2(self):
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
        test_fish, env, dyn = set_up_sim(n, source_id, attract=1, speed_up=1, angle=90)

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
        self.assertTrue(target_pos[0] < 0)
        self.assertTrue(target_pos[1] > 0)
        self.assertTrue(target_pos[3] > np.pi/2)

    def test_bv_align_move_3(self):
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
        test_fish, env, dyn = set_up_sim(n, source_id, attract=1, speed_up=1, angle=90)

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
        self.assertTrue(target_pos[0] < 0)
        self.assertTrue(target_pos[1] > 0)
        self.assertTrue(target_pos[3] > np.pi/2)


if __name__ == '__main__':
    unittest.main()