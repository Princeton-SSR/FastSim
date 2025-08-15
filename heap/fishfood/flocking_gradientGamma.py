"""Simulates a Bluebot. Add behavioral code here.

Leader-Follower simulation
Leader swim forward,  take turn by random choice
Follower moves towards the leader, using rel_pos (not LEDs)

commented out print statements
"""
from math import *
import numpy as np
import time
from fishfood.flocking_helper.flock_fct import *

# ONlY for testing
import os
import csv

Leader_initial = [-00,-2000, 0, pi * 2/4]

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

        # print("-- Fish run --")
        # print("id", self.id)
        # print("robots", robots)
        # print("rel_pos", rel_pos)
        # print("dist", dist)
        # print("leds", leds)
        # print("leds shape", leds.shape)


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

    def depth_ctrl_vision(self, r_move_g, dorsal_freq=1):
        """Vision-like depth control
        
        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        pitch_range = 1 # abs(pitch) below which dorsal fin is not controlled

        pitch = np.arctan2(r_move_g[2], sqrt(r_move_g[0]**2 + r_move_g[1]**2)) * 180 / pi

        if pitch > pitch_range:
            self.dorsal = dorsal_freq
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

    def move(self, robots, rel_pos, dist, duration):
        """Decision-making based on neighboring robots and corresponding move
        """

        leader_speed= 0.1 * a

        if self.id == 0: # leader

            self.caudal = leader_speed# a is from flock_fct.py, set the upper limit of phi_a function

            # self.pect_r = 0.01
            # leader turns by random choice
            choices = [[0, 0], [0, 0.3], [0.3, 0]] # [straight, turn left, turn right]
            weights = [1, 0.0, 0.]  # Higher weights for last two choices
            choice = np.random.choice(len(choices), p=weights)
            # print('leader move',choice, choices[choice])
            self.pect_l, self.pect_r = choices[choice]

            self.depth_ctrl_psensor(1000,1) # target depth, dorsal freq


        else: # alpha agents
            # Get the neighbours in range Rd


            masked_dist = dist.copy()
            masked_dist[self.id] = np.inf

            d = 2 * self.body_length # lattice scale (distance between a-agents)
            r = 5 * d # interaction range of a-agents
            neighbors = np.where(masked_dist < r)[0]

            # print("robot length", self.body_length, "lattice d", d, "interaction range r", r)

            # ## define 1 + leader(s) if neededs
            # leaders = [0,1]
            # leaders = [0]

            # # Find closest leader based on position comparison
            # distances_to_leaders = [np.linalg.norm(self.environment.pos[leader_id,:3] - self.environment.pos[self.id,:3]) for leader_id in leaders]
            # leader = leaders[np.argmin(distances_to_leaders)]
            leader = 0 # leader is always 0 in this case
            
            # phi_a_value = phi_a(self.environment.pos[self.id,:3], self.environment.pos[agent,:3], r_a, d_a)

            r_a = sigma_norm(r)
            d_a = sigma_norm(d)

            phi_a_value = 0

            u_i_alpha = np.zeros((3,)) # initialize interaction command
            g_i = np.zeros((3,)) # initialize gradient term
            c_i = np.zeros((3,)) # initialize consensus term

            for agent in neighbors:

                phi_a_value += phi_a(self.environment.pos[self.id,:3], self.environment.pos[agent,:3], r_a, d_a) # accumulate phi_a value

                gradient_term = phi_a(self.environment.pos[self.id,:3], self.environment.pos[agent,:3], r_a, d_a) * n_ij(self.environment.pos[self.id,:3], self.environment.pos[agent,:3]) 
                
                consensus_term = a_ij(self.environment.pos[self.id,:3], self.environment.pos[agent,:3], r_a) * (self.environment.vel[agent,:3] - self.environment.vel[self.id,:3])
                

                # consensus_term = 0 # disable consensus term for now

                g_i += c1_a * gradient_term
                c_i += c2_a * consensus_term

                u_ij = c1_a * gradient_term + c2_a * consensus_term

                u_i_alpha += u_ij

            if np.linalg.norm(g_i) > 0:
                g_i = g_i / np.linalg.norm(g_i)
            if np.linalg.norm(c_i) > 0:
                c_i = c_i / np.linalg.norm(c_i)

            u_i_alpha = 0.5 * g_i + 0.5 * c_i

     
            ## navigation term
            c2_g = 0 
            u_nav =  - c1_g * (self.environment.pos[self.id,:3] - self.environment.pos[leader, :3] ) - c2_g * (self.environment.vel[self.id,:3] - self.environment.vel[leader,:3]) # navigation term
            norm = np.linalg.norm(u_nav)
            u_nav = u_nav / norm if norm != 0 else u_nav  
            # u_nav = 0  

            u_i = 0.5 * g_i + 0.5 * c_i + 0.1 * u_nav # final move command

            u_i = 0.5 * g_i + 0.0 * c_i + 0.5 * u_nav # final move command
            
            move = u_i # move direction command
            magnitude = min(np.linalg.norm(u_i), leader_speed * 10)  # move magnitude, limit freq to 5 


            ####### debugging printing ########

            # Append data to file
            with open("/Users/di/Documents/GitHub/FastSim/heap/fishfood/flocking_helper/debug_test/phi_data.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.id, phi_a_value])

            print(" --------------------- in run/move ------------------------ ")
            print("self.id", self.id)
            
            # print("* phi_a", phi_a_value)5
            # print("distance to leader", np.linalg.norm(self.environment.pos[leader,:3] - self.environment.pos[self.id,:3]),"\n")

            print("* gradient_term", g_i, "magnitude", np.linalg.norm(g_i))
            # print("* consensus_term", c_i, "magnitude", np.linalg.norm(c_i))


            # print("*** u_i_alpha = gradient +  consensus", u_i_alpha, "magnitude of u_i_alpha:", np.linalg.norm(u_i_alpha))
                    
            print("* u_nav", u_nav, "magnitude of u_nav:", np.linalg.norm(u_nav))


            print('\n')
            print("u_i", u_i)

            print("move", move)
            print("magnitude", magnitude, "limited to ", leader_speed * 10)
            print(" --------------------- finish run/move ------------------------ ")
     

            # Global to Robot Transformation
            phi = self.environment.pos[self.id,3]
            r_T_g = self.environment.rot_global_to_robot(phi)
            r_move_g = r_T_g @ move

            
            self.home(r_move_g, magnitude)
            self.depth_ctrl_vision(r_move_g)



        ## ------ update control forces ------
        self.dynamics.update_ctrl(self.dorsal, self.caudal, self.pect_r, self.pect_l)
        target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)

        return (target_pos, self_vel)
    
