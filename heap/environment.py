"""Central data base keeping track of positions, velocities, relative positions, and distances of all simulated fishes
"""
import math
import random
import numpy as np
from scipy.spatial.distance import cdist
import sys
U_LED_DX = 86 # [mm] leds x-distance on BlueBot
U_LED_DZ = 86 # [mm] leds z-distance on BlueBot

class Environment():
    """Simulated fish environment
    
    Fish get their visible neighbors and corresponding relative positions and distances from here. Fish also update their own positions after moving in here. Environmental tracking data is used for simulation analysis.
    """

    def __init__(self, pos, vel, fish_specs, arena):
        # Arguments
        self.pos = pos # x, y, z, phi; [no_robots X 4]
        self.vel = vel # pos_dot
        self.v_range = fish_specs[0] # visual range, [mm]
        self.w_blindspot = fish_specs[1] # width of blindspot, [mm]
        self.r_sphere = fish_specs[2] # radius of blocking sphere for occlusion, [mm]
        self.n_magnitude = fish_specs[3] # visual noise magnitude, [% of distance]
        self.arena_size = arena # x, y, z
        
        # Parameters
        self.no_robots = self.pos.shape[0]
        self.no_states = self.pos.shape[1]

        # Initialize robot states
        self.init_states()

        # Initialize tracking
        self.init_tracking()

        # Initialize LEDs
        self.leds_pos = [np.zeros((3,3))]*self.no_robots # empty init, filled with update_leds() below
        for robot in range(self.no_robots):
            self.update_leds(robot)

    def log_to_file(self, filename):
        """Logs tracking data to file
        """
        np.savetxt('./logfiles/{}_data.txt'.format(filename), self.tracking, fmt='%.2f', delimiter=',')

    def init_tracking(self):
        """Initializes tracking
        """
        pos = np.reshape(self.pos, (1,self.no_robots*self.no_states))
        vel = np.reshape(self.vel, (1,self.no_robots*self.no_states))
        self.tracking = np.concatenate((pos,vel), axis=1)
        self.updates = 0

    def update_tracking(self):
        """Updates tracking after every fish took a turn
        """
        pos = np.reshape(self.pos, (1,self.no_robots*self.no_states))
        vel = np.reshape(self.vel, (1,self.no_robots*self.no_states))
        current_state = np.concatenate((pos,vel), axis=1)
        self.tracking = np.concatenate((self.tracking,current_state), axis=0)
    
    def update_leds(self, source_index):
        """ Updates the position of the three leds based on self.pos, which is the position of led1
        """
        pos = self.pos[source_index,:3]
        phi = self.pos[source_index,3]

        x1 = pos[0]
        x2 = x1
        x3 = x1 + math.cos(phi)*U_LED_DX

        y1 = pos[1]
        y2 = y1
        y3 = y1 + math.sin(phi)*U_LED_DX

        z1 = pos[2]
        z2 = z1 + U_LED_DZ
        z3 = z1

        self.leds_pos[source_index] = np.array([[x1, x2, x3],[y1, y2, y3],[z1, z2, z3]])

    def init_states(self):
        """Initializes fish positions and velocities
        """
        # Restrict initial positions to arena size
        self.pos[:,0] = np.clip(self.pos[:,0], 0, self.arena_size[0])
        self.pos[:,1] = np.clip(self.pos[:,1], 0, self.arena_size[1])
        self.pos[:,2] = np.clip(self.pos[:,2], 0, self.arena_size[2])

        # Initial relative positions
        a_ = np.reshape(self.pos, (1, self.no_robots*self.no_states))
        a = np.tile(a_, (self.no_robots,1))
        b = np.tile(self.pos, (1,self.no_robots))
        self.rel_pos = a - b # [4*no_robots X no_robots]

        # Initial distances
        self.dist = cdist(self.pos[:,:3], self.pos[:,:3], 'euclidean') # without phi; [no_robots X no_robots]

    def update_states(self, source_id, pos, vel): # add noise
        """Updates a fish state and affected realtive positions and distances
        """
        # Position and velocity
        self.pos[source_id,0] = np.clip(pos[0], 0, self.arena_size[0])
        self.pos[source_id,1] = np.clip(pos[1], 0, self.arena_size[1])
        self.pos[source_id,2] = np.clip(pos[2], 0, self.arena_size[2])
        self.pos[source_id,3] = pos[3]
        self.vel[source_id,:] = vel

        # Relative positions
        pos_others = np.reshape(self.pos, (1,self.no_robots*self.no_states))
        pos_self = np.tile(self.pos[source_id,:], (1,self.no_robots))
        rel_pos = pos_others - pos_self
        self.rel_pos[source_id,:] = rel_pos # row
        rel_pos_ = np.reshape(rel_pos, (self.no_robots, self.no_states))
        self.rel_pos[:,source_id*self.no_states:source_id*self.no_states+self.no_states] = -rel_pos_ # columns
        
        # Relative distances
        dist = np.linalg.norm(rel_pos_[:,:3], axis=1) # without phi
        self.dist[source_id,:] = dist
        self.dist[:,source_id] = dist.T

        # Update LEDs
        self.update_leds(source_id)

        # Update tracking
        self.updates += 1
        if self.updates >= self.no_robots:
            self.updates = 0
            self.update_tracking()

    def get_robots(self, source_id, visual_noise=False):
        """Provides visible neighbors and relative positions and distances to a fish
        """
        robots = set(range(self.no_robots)) # all robots
        robots.discard(source_id) # discard self

        rel_pos = np.reshape(self.rel_pos[source_id], (self.no_robots, self.no_states))

        self.visual_range(source_id, robots)
        self.blind_spot(source_id, robots, rel_pos)
        self.occlusions(source_id, robots, rel_pos)

        leds = self.calc_relative_leds(source_id, robots)

        if self.n_magnitude: # no overwrites of self.rel_pos and self.dist
            n_rel_pos, n_dist = self.visual_noise(source_id, rel_pos)
            return (robots, n_rel_pos, n_dist, leds)
        return (robots, rel_pos, self.dist[source_id], leds)

    def visual_range(self, source_id, robots):
        """Deletes fishes outside of visible range
        """
        conn_drop = 0.005
        
        candidates = robots.copy()
        for robot in candidates:
            d_robot = self.dist[source_id][robot]
            x = conn_drop * (d_robot - self.v_range)
            if x < -5:
                sigmoid = 1
            elif x > 5:
                sigmoid = 0
            else:
                sigmoid = 1 / (1 + math.exp(x))
            prob = random.random()

            if  sigmoid < prob:
                robots.remove(robot)

    def blind_spot(self, source_id, robots, rel_pos):
        """Omits fishes within the blind spot behind own body
        """
        r_blockage = self.w_blindspot/2

        phi = self.pos[source_id,3]
        phi_xy = [math.cos(phi), math.sin(phi)]
        mag_phi = np.linalg.norm(phi_xy)
        
        candidates = robots.copy()
        for robot in candidates:
            dot = np.dot(phi_xy, rel_pos[robot,:2])
            if dot < 0:
                d_robot = np.linalg.norm(rel_pos[robot,:2])

                angle = abs(math.acos(dot / (mag_phi * d_robot))) - math.pi / 2 # cos(a-b) = ca*cb+sa*sb = sa

                if  math.cos(angle) * d_robot < r_blockage:
                    robots.remove(robot)

    def occlusions(self, source_id, robots, rel_pos):
        """Omits invisible fishes occluded by others
        """


        rel_dist = self.dist[source_id]
        id_by_dist = np.argsort(rel_dist)

        n_valid = []
        for robot in id_by_dist[1:]:
            if not robot in robots:
                continue
            occluded = False
            d_robot = rel_dist[robot]
            if d_robot == 0: # "collision"
                continue
            coord_robot = rel_pos[robot,:3]

            for verified in n_valid:
                d_verified = rel_dist[verified]
                coord_verified = rel_pos[verified,:3]

                theta_min = math.atan(self.r_sphere / d_verified)
                theta = abs(math.acos(np.dot(coord_robot, coord_verified) / (d_robot * d_verified)))

                if theta < theta_min:
                    occluded = True
                    robots.remove(robot)
                    if not robots:
                        return
                    break

            if not occluded:
                n_valid.append(robot)

    def visual_noise(self, source_id, rel_pos):
        """Adds visual noise
        """
        magnitudes = self.n_magnitude * np.array([self.dist[source_id]]).T
        noise = magnitudes * (np.random.rand(self.no_robots, self.no_states) - 0.5) # zero-mean uniform noise
        n_rel_pos = rel_pos + noise
        n_dist = np.linalg.norm(n_rel_pos[:,:3], axis=1) # new dist without phi

        return (n_rel_pos, n_dist)

    def see_circlers(self, source_id, robots, rel_pos, sensing_angle):
        '''For circle formation
        '''
        phi = self.pos[source_id,3]
        phi_xy = [math.cos(phi), math.sin(phi)]
        mag_phi = np.linalg.norm(phi_xy)

        candidates = robots.copy()
        for robot in candidates:
            dot = np.dot(phi_xy, rel_pos[robot,:2])
            if dot > 0:
                d_robot = np.linalg.norm(rel_pos[robot,:2])

                angle = abs(math.acos(dot / (mag_phi * d_robot)))

                if (angle*180/math.pi) < (sensing_angle/2):
                    return True

        return False
    
    def count_left_right(self, source_id, robots, rel_pos, sensing_angle):
        '''counts the number of agents visible to the left and right of the agent. Math copied from BV utils/pts_left_right
        '''
        # self.pos as well as rel_pos are organized as [x,y,z,theta]
        phi = self.pos[source_id,3]
        candidates = robots.copy()
        n = len(candidates)

        # assuming that phi is in radians and oriented starting at the positive x-axis. This assumption is now verified
        perp = phi - (np.pi/2) # 90 degrees clockwise 
        unit_perp = np.array([np.cos(perp), np.sin(perp)])
        dot_perp = np.dot(np.full((n,2),unit_perp), rel_pos[list(candidates),:2].T) # dot product between the current orientation and each relative position vector
        diag_perp = np.diagonal(dot_perp) # I'm not sure why this step is necessary, but the dot product returns the values along the diagonal of a large square matrix, this pulls them out into a 1D array
        sign_perp = np.sign(diag_perp) # 1 means the light is to the right of the robot, -1 to the left, 0 directly ahead
        right_count = len(sign_perp[sign_perp>0])
        left_count = len(sign_perp[sign_perp<0])

        return left_count, right_count

    def rot_global_to_robot(self, phi):
        """Rotate global coordinates to robot coordinates. Used before simulation of dynamics.
        """
        return np.array([[math.cos(phi), math.sin(phi), 0], [-math.sin(phi), math.cos(phi), 0], [0, 0, 1]])

    def rot_robot_to_global(self, phi):
        """Rotate robot coordinates to global coordinates. Used after simulation of dynamics.
        """
        return np.array([[math.cos(phi), -math.sin(phi), 0], [math.sin(phi), math.cos(phi), 0], [0, 0, 1]])

    def calc_reflections(self, leds_list):
        """Calculates the position of the reflected leds
        """
        refl_list = []
        for led in leds_list:
            if led[2] > 10: # at least 10 mm below surface to have a reflection
                refl = led + np.array([0,0, -2*led[2]])
                refl_list.append(refl)
        return refl_list

    def calc_relative_leds(self, source_id, robots):
        """Calculates the relative position of all detectable leds and adds their reflection if add_reflections boolean is set to True
        """
        if not robots:
            return np.empty((3,0))

        add_reflections = True
        all_blobs = np.empty((3,0))
        
        leds = []
        for robot in robots:
            leds.append(self.leds_pos[robot])

        leds_list = list(np.transpose(np.hstack(leds)))
        if add_reflections:
            refl_list = self.calc_reflections(leds_list)
            leds_list = leds_list + refl_list

        my_pos = self.pos[source_id,:3]
        my_phi = self.pos[source_id,3]
        R = self.rot_global_to_robot(my_phi)

        for led in leds_list:
            relative_coordinates = R @ ((led - my_pos)[:, np.newaxis])
            relative_coordinates /= np.linalg.norm(relative_coordinates) # normalize from xyz to pqr
            all_blobs = np.append(all_blobs, relative_coordinates, axis=1)

        p = np.random.permutation(np.shape(all_blobs)[1]) # mix up into random order
        return all_blobs[:,p]
