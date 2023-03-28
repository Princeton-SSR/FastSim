import numpy as np
import unittest

from environment import *
import fishfood.bv_align as bv
# from environment import Environment
from dynamics import Dynamics
from unit_test_utils import *

class TestEnvironmentMethods(unittest.TestCase):

    def test_environment_occlusions_1(self):
        """
        Basic test of 2 agents with no possible occlusions
        """

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

        # before we run occlusions we expect the robot list to include all robots
        new_robots = robots.copy()

        test_fish.environment.occlusions(source_id, new_robots, rel_pos)

        self.assertEqual(robots,new_robots)

    def test_environment_occlusions_2(self):
        """
        Basic test with 1 trivial occlusion
        """
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
        env.pos = np.array([[0,0,100,np.pi/2],[50,50,100,np.pi/2],[100,100,100,np.pi/2]])
        # robots is a list of indices (excluding self)
        robots = np.arange(1,n)
        # rel_pos is the relative positions as a list of [x,y,z,theta]
        # let self be at [0,0,np.pi/2]
        rel_pos = np.array([[0,0,0,0],[100,100,0,0]])
        
        # robots, rel_pos, dist, leds = test_fish.environment.get_robots(source_id=0)

        # before we run occlusions we expect the robot list to include all robots
        new_robots = robots.copy()

        test_fish.environment.occlusions(source_id, new_robots, rel_pos)

        self.assertEqual(new_robots,np.array([1]))



if __name__ == '__main__':
    unittest.main()