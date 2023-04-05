"""Simulates a Bluebot. Add behavioral code here.
"""
from math import *
import numpy as np
import random
import time


class Fish():
    """Bluebot instance
    """
    
    def __init__(self, my_id, dynamics, environment, attract, speed_up):
        # Arguments
        self.id = my_id
        self.dynamics = dynamics
        self.environment = environment
        self.attract = attract
        self.speed_up = speed_up

        # Bluebot features
        self.body_length = 130

        # Fins
        self.caudal = 0
        self.dorsal = 0
        self.pect_r = 0
        self.pect_l = 0

        # Behavior specific
        self.target_depth = random.randint(250, 1170-250)

    def run(self, duration):
        """(1) Get neighbors from environment, (2) move accordingly, (3) update your state in environment
        """
        attract = 1
        speed_up = 1
        robots, rel_pos, dist, leds = self.environment.get_robots(self.id)
        target_pos, vel = self.move(robots, rel_pos, dist, duration, attract, speed_up)
        self.environment.update_states(self.id, target_pos, vel)

    def lj_force(self, robots, rel_pos, dist, r_target):
        """lj_force derives the Lennard-Jones potential and force based on the relative positions of all neighbors and the desired self.target_dist to neighbors. The force is a gain factor, attracting or repelling a fish from a neighbor. The center is a point in space toward which the fish will move, based on the sum of all weighted neighbor positions.
        """
        a = 12
        b = 6
        epsilon = 1 # depth of potential well, V_LJ(r_target) = epsilon
        gamma = 10 # force gain
        r_const = r_target + 2 * self.body_length

        center = np.zeros((3,))
        n = len(robots)

        for robot in robots:
            r = min(dist[robot], r_const)
            f_lj = -gamma*epsilon/r * (a*(r_target/r)**a - 2*b*(r_target/r)**b)
            center += f_lj * rel_pos[robot,:3]

        center /= n
        magn = np.linalg.norm(center) # normalize
        center /= magn # normalize

        return (center, magn)

    def depth_ctrl_vision(self, r_move_g):
        """Vision-like depth control
        
        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        pitch_range = 1 # abs(pitch) below which dorsal fin is not controlled

        pitch = np.arctan2(r_move_g[2], sqrt(r_move_g[0]**2 + r_move_g[1]**2)) * 180 / pi

        if pitch > pitch_range:
            self.dorsal = 1
        elif pitch < -pitch_range:
            self.dorsal = 0

    def depth_ctrl_psensor(self, target_depth):
        """Pressure-sensor-like depth control
        
        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        my_depth = self.environment.pos[self.id,2]

        if my_depth > target_depth:
            self.dorsal = 0
        else:
            self.dorsal = 1

    def home(self, r_move_g, magnitude):
        """Homing behavior. Sets fin controls to move toward a desired goal location.
        
        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
            magnitude (TYPE): Description
        """
        caudal_range = 35 # abs(heading) below which caudal fin is switched on
        freq_c = min(0.5 + 1/250 * magnitude, 1)

        heading = np.arctan2(r_move_g[1], r_move_g[0]) * 180 / pi

        # target behind
        if heading > 155 or heading < -155:
            self.caudal = 0
            self.pect_r = 1.5
            self.pect_l = 1.5

        # target in front
        elif heading < 10 and heading > -10:
            self.pect_r = 0
            self.pect_l = 0
            self.caudal = freq_c

        # target to the right
        elif heading > 10:
            freq_l = 0.5 + 1 * abs(heading) / 155
            self.pect_l = freq_l
            self.pect_r = 0

            if heading < caudal_range:
                self.caudal = freq_c
            else:
                self.caudal = 0

        # target to the left
        elif heading < -10:
            freq_r = 0.5 + 1 * abs(heading) / 155
            self.pect_r = freq_r
            self.pect_l = 0

            if heading > -caudal_range:
                self.caudal = freq_c
            else:
                self.caudal = 0

    def circling(self, robots, rel_pos):
        sensing_angle = 25 #deg

        if not robots:
            self.pect_l = 0
            self.pect_r = 0.5
            self.caudal = 0.1
            return
        
        someone = self.environment.see_circlers(self.id, robots, rel_pos, sensing_angle)

        if someone:
            self.pect_r = 0
            self.pect_l = 0.5
            self.caudal = 0.1
        else:
            self.pect_l = 0
            self.pect_r = 0.5
            self.caudal = 0.1     

    def bv_align(self, robots, rel_pos):

        # attract = 1 or 0
        # speed_up  =1 or 0
        # sensing_angle = 90 #deg

        if not robots:
            self.pect_l = 0
            self.pect_r = 0
            self.caudal = 0
            return

        left_count, right_count = self.environment.count_left_right(self.id, robots, rel_pos)
        # print("left_count = " + str(left_count) + ", right_count = " + str(right_count))

        if left_count > right_count:
            self.pect_l = 0.2*left_count # attract
            self.pect_r = 0             # attract 
            self.caudal = 0.2*left_count # speed_up
        elif left_count < right_count:
            self.pect_l = 0 # attract
            self.pect_r = 0.2*right_count # attract
            self.caudal = 0.2*right_count # speed up
        else: 
            self.pect_l = 0
            self.pect_r = 0
            self.caudal = 0

    def bv_align_paramterized(self, robots, rel_pos, attract, speed_up, influence=.2):
        
        
        # attract and speed_up must take binary values of 0 or 1 
        assert (attract == 0 or attract == 1)
        assert (speed_up == 0 or speed_up == 1)
        
        if not robots:
            self.pect_l = 0
            self.pect_r = 0
            self.caudal = 0
            return

        left_count, right_count = self.environment.count_left_right(self.id, robots, rel_pos)
        # print("left_count = " + str(left_count) + ", right_count = " + str(right_count))

        # speed toward
        if attract == 1 and speed_up == 1:
            if left_count > right_count:
                self.pect_l = influence*left_count # attract
                self.pect_r = 0             # attract 
                self.caudal = influence*left_count # speed_up
            elif left_count < right_count:
                self.pect_l = 0 # attract
                self.pect_r = influence*right_count # attract
                self.caudal = influence*right_count # speed up
            else: 
                self.pect_l = 0
                self.pect_r = 0
                self.caudal = 0
        
        # speed away 
        if attract == 0 and speed_up == 1:
            if left_count > right_count:
                self.pect_l = 0 # attract
                self.pect_r = influence*left_count # attract
                self.caudal = influence*left_count # speed_up
            elif left_count < right_count:
                self.pect_l = influence*right_count # attract
                self.pect_r = 0
                self.caudal = influence*right_count # speed up
            else: 
                self.pect_l = 0
                self.pect_r = 0
                self.caudal = 0
        
        # slow toward
        if attract == 1 and speed_up == 0:
            if left_count > right_count:
                self.pect_l = influence*left_count # attract
                self.pect_r = 0             # attract 
                self.caudal = 1-influence*left_count # speed_up
            elif left_count < right_count:
                self.pect_l = 0 # attract
                self.pect_r = influence*right_count # attract
                self.caudal = 1-influence*right_count # speed up
            else: 
                self.pect_l = 0
                self.pect_r = 0
                self.caudal = 0

        # slow away 
        #TODO: The way we're going about slowing down is wrong. This is actually speeding things up. 
        if attract == 0 and speed_up == 0:
            if left_count > right_count:
                self.pect_l = 0 # attract
                self.pect_r = influence*left_count # attract
                self.caudal = 1-influence*left_count # speed_up
            elif left_count < right_count:
                self.pect_l = influence*right_count # attract
                self.pect_r = 0
                self.caudal = 1-influence*right_count # speed up
            else: 
                self.pect_l = 0
                self.pect_r = 0
                self.caudal = 0

    def move(self, robots, rel_pos, dist, duration, attract, speed_up):
        """Decision-making based on neighboring robots and corresponding move
        """
        if not robots: # no robots, continue with ctrl from last step
            target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)
            return (target_pos, self_vel)

        # self.circling(robots, rel_pos)
        self.bv_align_paramterized(robots, rel_pos, attract, speed_up)
        self.depth_ctrl_psensor(self.target_depth)

        self.dynamics.update_ctrl(self.dorsal, self.caudal, self.pect_r, self.pect_l)

        target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)

        return (target_pos, self_vel)
    
        # TODO: figure out which pieces are unnecessary for this algorithm, and set up way to comment them out for calculating relative loop iteration time

    