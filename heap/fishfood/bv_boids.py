"""Simulates a Bluebot. Add behavioral code here.
"""
from math import *
import numpy as np
import random
import time
import pdb


class Fish():
    """Bluebot instance
    """
    
    def __init__(self, my_id, dynamics, environment, attract=1, speed_up=1, sensing_angle=170):
        # Arguments
        self.id = my_id
        self.dynamics = dynamics
        self.environment = environment
        self.attract = attract
        self.speed_up = speed_up
        self.sensing_angle = sensing_angle

        # Bluebot features
        self.body_length = 130

        # Fins
        self.caudal = 0
        self.dorsal = 0
        self.pect_r = 0
        self.pect_l = 0

        # Behavior specific
        self.target_depth = random.randint(200, 300)
        self.sensing_angle = sensing_angle

    def run(self, duration):
        """(1) Get neighbors from environment, (2) move accordingly, (3) update your state in environment
        """
        attract = 1
        speed_up = 0
        robots, rel_pos, dist, leds = self.environment.get_robots(self.id)
        target_pos, vel = self.move(robots, rel_pos, dist, duration, attract, speed_up)
        self.environment.update_states(self.id, target_pos, vel)

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

    def bv_align_paramterized(self, robots, rel_pos, dist, attract, speed_up, influence=.2):
        # attract and speed_up must take binary values of 0 or 1 
        assert (attract == 0 or attract == 1)
        assert (speed_up == 0 or speed_up == 1)
        
        if not robots:
            self.pect_l = 0
            self.pect_r = 0
            self.caudal = 0
            return

        left_count, right_count, ind_left, ind_right = self.environment.count_left_right(self.id, robots, rel_pos)
        # print("left_count = " + str(left_count) + ", right_count = " + str(right_count))

        # split by distances, max distance and repel distance
        # rep_distance = self.environment.v_range//4
        # # distances = np.linalg.norm(rel_pos[:,:3], axis=1)
        # ind_rep = np.where(dist < rep_distance)[0]
        # # pdb.set_trace()

        # ind_rep_right = np.intersect1d(ind_rep, ind_right)
        # # ind_attr_right = np.setdiff1d(ind_right, ind_rep_right)
        # ind_rep_left = np.intersect1d(ind_rep, ind_left)
        # # ind_attr_left = np.setdiff1d(ind_left, ind_rep_left)

        # right_count_tot = right_count - len(ind_rep_right) + len(ind_rep_left)
        # left_count_tot = left_count - len(ind_rep_left) + len(ind_rep_right)
        

        # right_count = right_count_tot
        # left_count = left_count_tot

        # speed toward
        if attract == 1 and speed_up == 1:
            if left_count > right_count:
                self.pect_l = 4*influence*left_count # attract
                self.pect_r = 0             # attract 
                self.caudal = influence*left_count # speed_up
            elif left_count < right_count:
                self.pect_l = 0 # attract
                self.pect_r = 4*influence*right_count # attract
                self.caudal = influence*right_count # speed up
            else: 
                self.pect_l = 0
                self.pect_r = 0
                self.caudal = 0

        # slow toward
        if attract == 1 and speed_up == 0:
            if left_count > right_count:
                turn_l = 4*influence*left_count # attract
                turn_r = 0             # attract 
                forward = 1-influence*left_count # speed_up
            elif left_count < right_count:
                turn_l = 0 # attract
                turn_r = 4*influence*right_count # attract
                forward = 1-influence*right_count # speed up
            else: 
                turn_l = 0
                turn_r = 0
                forward = 0

        # clip
        if forward > 1:
            forward = 1
        elif forward < 0:
            forward = 0
        if turn_l > 1:
            turn_l = 1
        elif turn_l < 0:
            turn_l = 0
        if turn_r > 1:
            turn_r = 1
        elif turn_r < 0:
            turn_r = 0

        # then cast to fins, this cancels out the reverse motion caused by the pect fins
        self.pect_l = turn_l
        self.pect_r = turn_r
        self.caudal = forward + 0.5*(turn_l + turn_r)

        # clip again
        if self.caudal > 1:
            self.pect_l = self.pect_l/self.caudal
            self.pect_r = self.pect_r/self.caudal
            self.caudal = 1
        if self.pect_l > 1:
            self.pect_l = 1
            self.caudal = forward + 0.5
        if self.pect_r > 1:
            self.pect_r = 1
            self.caudal = forward + 0.5
        


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


        # clip
        # if(self.pect_l > 1):
        #     self.pect_l = 1
        # elif(self.pect_l < 0):
        #     self.pect_l = 0

        # if(self.pect_r > 1):
        #     self.pect_r = 1
        # elif(self.pect_r < 0):
        #     self.pect_r = 0

        # if(self.caudal > 1):
        #     self.caudal = 1
        # elif(self.caudal < 0):
        #     self.caudal = 0

        # print(str(self.pect_l) + ", " + str(self.pect_r) + ", " + str(self.caudal))

    def move(self, robots, rel_pos, dist, duration, attract, speed_up):
        """Decision-making based on neighboring robots and corresponding move
        """
        if not robots: # no robots, continue with ctrl from last step
            target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)
            return (target_pos, self_vel)

        # self.circling(robots, rel_pos)
        new_robots = self.environment.angle_threshold(self.id, robots, rel_pos, self.sensing_angle)
        # print(new_robots)
        self.bv_align_paramterized(new_robots, rel_pos, dist, attract, speed_up, influence=0.2)
        self.depth_ctrl_psensor(self.target_depth)

        self.dynamics.update_ctrl(self.dorsal, self.caudal, self.pect_r, self.pect_l)

        target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)

        return (target_pos, self_vel)
    
        # TODO: figure out which pieces are unnecessary for this algorithm, and set up way to comment them out for calculating relative loop iteration time

    