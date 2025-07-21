"""Simulates a Bluebot. Add behavioral code here.

Leader-Follower simulation
Leader swim forward 
Follower moves towards the leader's SIDE, using rel_pos (not LEDs)

"""
from math import *
import numpy as np
import time

Leader_initial = [-000,-2000, 0, pi * 2/4]
group_number = 3

class Fish():
    """Bluebot instance
    """
    
    def __init__(self, my_id, dynamics, environment):
        # Arguments
        self.id = my_id
        self.dynamics = dynamics
        self.environment = environment

        # Bluebot features
        self.body_length = 130

        # Fins
        self.caudal = 0
        self.dorsal = 0
        self.pect_r = 0
        self.pect_l = 0


    def run(self, duration):
        """(1) Get neighbors from environment, (2) move accordingly, (3) update your state in environment
        """
        robots, rel_pos, dist, leds = self.environment.get_robots(self.id)
        target_pos, vel = self.move(robots, rel_pos, dist, duration)
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

    def depth_ctrl_psensor(self, target_depth, dorsal_freq):
        """Pressure-sensor-like depth control
        
        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        depth = self.environment.pos[self.id,2]

        if depth < target_depth:
            self.dorsal = dorsal_freq
        else:
            self.dorsal = 0

    def home(self, r_move_g, magnitude):
        """Homing behavior. Sets fin controls to move toward a desired goal location.
        
        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
            magnitude (TYPE): Description
        """
        caudal_range = 35 # abs(heading) below which caudal fin is switched on
        # freq_c = min(0.5 + 1/250 * magnitude, 1)
        freq_c = magnitude

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

    # def move(self, robots, rel_pos, dist, duration):
    #     """Decision-making based on neighboring robots and corresponding move
    #     Centralized Control -- get access to leader's position from self.environment.pos
    # 
    # """

    #     if self.id == 0: # leader
    #         self.caudal = 1
    #         self.pect_r = 0.
    #         self.depth_ctrl_psensor(1500,1) # target depth, dorsal freq

    #     else:
    #         # Calculate offset position behind and left of leader
    #         offset_distance = 300.0  # distance to maintain from leader
    #         offset_angle = -np.pi/4  # 45 degrees to the left
            
    #         # Get leader's orientation
    #         leader_phi = self.environment.pos[0,3]
            
    #         # Calculate offset position in polar coordinates relative to leader's heading
    #         offset = np.array([
    #             -offset_distance * np.cos(leader_phi - offset_angle),  # x offset
    #             -offset_distance * np.sin(leader_phi - offset_angle),  # y offset
    #             0                                                      # z offset
    #         ])
            
    #         # Desired position is leader's position plus offset
    #         move = rel_pos[0,:3] + offset
    #         magnitude = 2

    #         # Global to Robot Transformation
    #         phi = self.environment.pos[self.id,3]
    #         r_T_g = self.environment.rot_global_to_robot(phi)

    #         r_move_g = r_T_g @ move

    #         self.depth_ctrl_vision(r_move_g)
    #         self.home(r_move_g, magnitude)

    #     self.dynamics.update_ctrl(self.dorsal, self.caudal, self.pect_r, self.pect_l)
    #     target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)

    #     return (target_pos, self_vel)
    
    def move(self, robots, rel_pos, dist, duration):



        following_angle = -np.pi/4

        if self.id == 0: # leader
            self.caudal = 0.5
            # self.pect_r = 0.1
            self.depth_ctrl_psensor(1500,1)
        elif self.id % group_number == 0: # follower GROUP 1
            """ follow the leader - obtain leader position from environment """
            # fix following
            offset_distance = 100.0 + self.id * 100 
            offset_angle = following_angle
            # Get leader's orientation
            leader_phi = self.environment.pos[0,3]
            offset = np.array([
                -offset_distance * np.cos(leader_phi - offset_angle),  # x offset
                -offset_distance * np.sin(leader_phi - offset_angle),  # y offset
                0            # z offset
            ])
    
            # Desired position is leader's position plus offset
            move = rel_pos[0,:3] + offset
            magnitude = 1.5
            # magnitude = np.tanh(np.linalg.norm(move)/300)

            # Global to Robot Transformation
            phi = self.environment.pos[self.id,3]
            r_T_g = self.environment.rot_global_to_robot(phi)

            r_move_g = r_T_g @ move

            self.depth_ctrl_vision(r_move_g)
            self.home(r_move_g, magnitude)

        elif self.id % group_number == 1: # follower GROUP 2
            """ follow the leader - obtain leader position from environment """
            # fix following
            offset_distance = 100.0 + self.id * 100 
            offset_angle = - following_angle
            # Get leader's orientation
            leader_phi = self.environment.pos[0,3]
            offset = np.array([
                -offset_distance * np.cos(leader_phi - offset_angle),  # x offset
                -offset_distance * np.sin(leader_phi - offset_angle),  # y offset
                0            # z offset
            ])
    
            # Desired position is leader's position plus offset
            move = rel_pos[0,:3] + offset
            magnitude = min(3, 1.5 + self.id)
            magnitude = 1.5

            # magnitude = np.tanh(np.linalg.norm(move)/300)

            # print("------------magnitude------------------")
            # print(magnitude)
            # print("------------------------------")

            # Global to Robot Transformation
            phi = self.environment.pos[self.id,3]
            r_T_g = self.environment.rot_global_to_robot(phi)

            r_move_g = r_T_g @ move

            self.depth_ctrl_vision(r_move_g)
            self.home(r_move_g, magnitude)

        else: # to be tested
            """ follow the closest robot - not working so far
            only works for leader + 1 robots
               """

            # print("seld id", self.id)
            # print("dist", dist)
            # print("dist[self.id]", dist[self.id])
            # print("dist[0]", dist[0])

            masked_dist = dist.copy()
            masked_dist[self.id] = np.inf
            nn = np.argmin(masked_dist)
            # print("nearest neighbor", nn)

            k = 1
            nn = np.argpartition(masked_dist, k)[:k]


            # print("nearest neighbor (s)", nn)

            leader_id = nn[0]

            leader_phi = self.environment.pos[leader_id,3]  
            # leader_phi = rel_pos[leader_id,3] + self.environment.pos[self.id,3] # can also be written as this, but still reply on env.pos
     
            # Calculate offset position behind and left
            offset_distance = 300.0
            offset_angle = np.pi/2  # 45 degrees
            
            offset = np.array([
                -offset_distance * np.cos(leader_phi - offset_angle),
                -offset_distance * np.sin(leader_phi - offset_angle),
                0
            ])
            
            move = rel_pos[leader_id,:3] + offset
            magnitude = 2
    
            phi = self.environment.pos[self.id,3]
            r_T_g = self.environment.rot_global_to_robot(phi)
            r_move_g = r_T_g @ move
    
            self.depth_ctrl_vision(r_move_g)
            self.home(r_move_g, magnitude)
    
        self.dynamics.update_ctrl(self.dorsal, self.caudal, self.pect_r, self.pect_l)
        target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)
        return (target_pos, self_vel)
