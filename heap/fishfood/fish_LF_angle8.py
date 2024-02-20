"""Simulates a Bluebot. Add behavioral code here.
Leader-Follower simulation
Leader stop/swim forward 
follower follows on the right (set target on leader's right)
(follower use only local LED info)

Expand to more than 2 agents, 1 leader + N followers (debugging)

Running instrucion:
go to enviornment python 3.6
$$ conda activate myenv36
Go to 
$$ cd */FastSim/heap
Run python (algo file after simulation.py) 
$$ python3 simulation.py dispersion

"""
from math import *
import numpy as np
import time
import warnings

U_LED_DX = 86 # [mm] leds x-distance on BlueBot
U_LED_DZ = 86 # [mm] leds z-distance on BlueBot
N_fish = 7
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

        # print('robot id', self.id)

        # (1) Get neighbors from environment
        robots, rel_pos, dist, leds, abs_leds = self.environment.get_robots(self.id)

        # (2) Move according to the self.move algorithm below, perform Eular integration 
        target_pos, vel = self.move(robots, rel_pos, dist, leds, abs_leds, duration)

        # (3) Update the environment
        self.environment.update_states(self.id, target_pos, vel)

    def depth_ctrl_vision(self, r_move_g):
        """Vision-like depth control
        
        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        pitch_range = 0 # abs(pitch) below which dorsal fin is not controlled
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
        # print(" current depth is, ", depth)
        
    

        if depth < target_depth:
            self.dorsal = dorsal_freq
        else:
            self.dorsal = 0

        # print("dorsal control, ", self.dorsal)

    def home(self, r_move_g, magnitude):
        """Homing behavior. Sets fin controls to move toward a desired goal location.
        
        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
            magnitude (TYPE): Description
        """
        caudal_range = 20 # abs(heading) below which caudal fin is switched on
        # freq_c = min(0.5 + 1/250 * magnitude, 1)
        freq_c = magnitude

        behind_threshold = 155
        front_threshold = 2

        heading = np.arctan2(r_move_g[1], r_move_g[0]) * 180 / pi
        # print("in fish_LF/home")
        # print("heading is", heading)

        # target behind
        if heading > behind_threshold or heading < -behind_threshold:
            # print("target behind")
            # spin clock-wise
            self.caudal = 0
            # self.pect_r = 1.5
            self.pect_l = 0.5

        # target in front
        elif heading < front_threshold and heading > -front_threshold:
            # print("target in front")
            self.pect_r = 0
            self.pect_l = 0
            self.caudal = freq_c

        # target to the right
        elif heading > front_threshold:
            # print("target to the right")
            freq_l = 0.5 + 1 * abs(heading) / behind_threshold
            self.pect_l = freq_l
            self.pect_r = 0

            if heading < caudal_range:
                self.caudal = freq_c
            else:
                self.caudal = 0

        # target to the left
        elif heading < -front_threshold:
            # print("target to the left")
            freq_r = 0.5 + 1 * abs(heading) / behind_threshold
            self.pect_r = freq_r
            self.pect_l = 0

            if heading > -caudal_range:
                self.caudal = freq_c
            else:
                self.caudal = 0

    def wait(self, pect):
        # swim backward
        self.caudal = 0
        self.pect_r = pect
        self.pect_l = pect

    def stop(self):
        """
        fully stopped (no fin actuated except dorsal)
        """
        self.caudal = 0
        self.pect_r = 0
        self.pect_l = 0   

    def forward(self, magnitude):
        self.caudal = magnitude
        self.pect_r = 0
        self.pect_l = 0
    
    def spin(self, caudal, pect, cw = True):
        self.caudal = caudal
        if cw:
            self.pect_r = 0
            self.pect_l = pect
        else: 
            self.pect_r = pect
            self.pect_l = 0         

    def translate(self,  pos, vector, keep_angle,distance):        
        # Rotation angle in radians
        angle = np.radians(keep_angle)

        # Rotation matrix for a 90-degree rotation about the z-axis
        # (rot_robot_to_global)
        rotation_matrix_z = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        # Calculate the Euclidean norm
        norm = np.linalg.norm(vector)

        # Normalize the vector
        normalized_vector = vector / norm



        new_pos = pos + rotation_matrix_z@normalized_vector * distance

        

        # print("in fish_LF/translate")
        # print("pos")
        # print(pos)

        # print("translate direction")
        # print("normalized vector")
        # print(normalized_vector)


        # print("rotation_matrix_z@normalized_vector")
        # print(rotation_matrix_z@normalized_vector)

        # print("rotation_matrix_z@normalized_vector* distance")
        # print(rotation_matrix_z@normalized_vector * distance)


        # print()

        return new_pos

    def calc_relative_angles(self, blobs): #copied and adapted from BlueSwarm Code "avoid_duplicates_by_angle" #pw split this up in env and fish part?
        """Use right and left cameras just up to the xz-plane such that the overlapping camera range disappears and there are no duplicates.

        Returns:
            tuple: all_blobs (that are valid, i.e. not duplicates) and their all_angles
        """
        angles = np.empty(0)
        for i in range(np.shape(blobs)[1]):
            led = blobs[:,i]
            angle = np.arctan2(led[1], led[0])
            angles = np.append(angles, angle)

        return angles #angles in rad!

    def calc_relative_pitch(self, blobs): #copied and adapted from BlueSwarm Code "avoid_duplicates_by_angle" #pw split this up in env and fish part?
        """Use right and left cameras just up to the xz-plane such that the overlapping camera range disappears and there are no duplicates.

        Returns:
            tuple: all_blobs (that are valid, i.e. not duplicates) and their all_angles
        """

        

        pitches = np.empty(0)
        for i in range(np.shape(blobs)[1]):
            led = blobs[:,i]
            pitch = np.arctan2(led[2], sqrt(led[0]**2 + led[1]**2)) * 180 / pi
            pitches = np.append(pitches, pitch)

        return pitches #angles in deg!

    def parsing(self, blobs): #copied and adapted from BlueSwarm Code "avoid_duplicates_by_angle" #pw split this up in env and fish part?
        """sort LEDs and output led 1, 2, 3 in order

        Returns:
            tuple: all_blobs (that are valid, i.e. not duplicates) and their all_angles
        """

        # print("in fish_LF/parsing")
        # print("input leds")

        # print(" leds shape", blobs.shape)
        # print(blobs)


        angles = np.empty(0)
        for i in range(np.shape(blobs)[1]):
            led = blobs[:,i]
            angle = np.arctan2(led[1], led[0])
            angles = np.append(angles, angle)
        
        pitches = np.empty(0)
        for i in range(np.shape(blobs)[1]):
            led = blobs[:,i]
            pitch = np.arctan2(led[2], sqrt(led[0]**2 + led[1]**2)) * 180 / pi
            pitches = np.append(pitches, pitch)

        pitch_sort_ind = np.argsort(pitches)



        err = 1 * pi/180
        if abs(angles[pitch_sort_ind[0]] - angles[pitch_sort_ind[2]]) < err:
            # print("cond1", pitch_sort_ind)
            pass
            
        elif abs(angles[pitch_sort_ind[1]] - angles[pitch_sort_ind[2]]) < err:
            pitch_sort_ind[0], pitch_sort_ind[1] = pitch_sort_ind[1], pitch_sort_ind[0]
            # print("cond2", pitch_sort_ind)
        # else: 
        #     print("angle between led 1 and led 2 too big")
        #     print("angles, ", angles)
        #     print("angle difference, " , abs(angles[pitch_sort_ind[0]] - angles[pitch_sort_ind[2]]), abs(angles[pitch_sort_ind[1]] - angles[pitch_sort_ind[2]]) )
        #     print("angle difference, " , degrees(abs(angles[pitch_sort_ind[0]] - angles[pitch_sort_ind[2]])), degrees(abs(angles[pitch_sort_ind[1]] - angles[pitch_sort_ind[2]])) )
        
        leds_132 = blobs[:,pitch_sort_ind] # re-arrange leds in the order of 1,3,2

        
        p = [0,2,1]
        leds_pqr = leds_132[:,p]  # re-arrange leds in the order of 1,2,3
        
        # print("parsed leds (pqr)")
        # print(leds_pqr)
        return leds_pqr #angles in deg!

    def _pqr_to_xyz(self, pqr): #twoblob

        p1 = pqr[0, 0]
        q1 = pqr[1, 0]
        r1 = pqr[2, 0]
        p2 = pqr[0, 1]
        q2 = pqr[1, 1]
        r2 = pqr[2, 1]

        if r2 < r1:
            ptemp = p1
            qtemp = q1
            rtemp = r1
            p1 = p2
            q1 = q2
            r1 = r2
            p2 = ptemp
            q2 = qtemp
            r2 = rtemp

        delta = U_LED_DZ

        xyz = np.empty([3,2])
        if abs(r2*p1 -r1*p2) < 0.0001:
            #print("pqr div by zero risk",pqr)
            d1 = 1
        else:
            d1 = p2 * delta/(r2*p1 -r1*p2)

        if abs(r2) < 0.0001:
            # print("pqr div by zero risk",pqr)
             d2 = d1
        else:
             d2 = (d1*r1 + delta)/r2

        xyz[:,0] = d1 * pqr[:,0]
        xyz[:,1] = d2 * pqr[:,1]

        return xyz
    
    def _pqr_3_to_xyz(self, xyz_1_2, pqr_3):
        """Converts blob3 from pqr into xyz by finding the scale factor
        Args:
        pqr_3 (float array 3x1): pqr coordinates of blob3
        xyz_1_2 (float array 3x2): xyz coordinates of blob1,2

        Returns:
        xyz (float array 3x3): xyz coordinates of the three blobs
        """

        x1 = xyz_1_2[0, 0]
        y1 = xyz_1_2[1, 0]
        z1 = xyz_1_2[2, 0]
        p3 = pqr_3[0]
        q3 = pqr_3[1]
        r3 = pqr_3[2]

        delta = U_LED_DX

        a = p3**2 + q3**2
        b = -2 * (x1*p3 + y1*q3)
        c = x1**2 + y1**2 - delta**2

        sqrt_pos = max(b**2 - 4 * a * c, 0) #pw preventing negative sqrt, if it is too far off, the blob will be discarded later by led_hor_dist
        d_plus = (-b + sqrt(sqrt_pos)) / (2 * a)
        d_minus = (-b - sqrt(sqrt_pos)) / (2 * a)

        diff_z_plus = abs(z1 - d_plus*pqr_3[2])
        diff_z_minus = abs(z1 - d_minus*pqr_3[2])
        #print("diff_z_plus,diff_z_minus",diff_z_plus,diff_z_minus)
        wrong_solution_prob = 0.0 #to simulate wrong solution, set to zero in normal case!!
        if (diff_z_plus < diff_z_minus): #choose solution for which led 3 is closer to same vertical height to led1
            xyz_3 = d_plus * pqr_3
            #simulate wrong solution
            if np.random.rand(1) < wrong_solution_prob:
                xyz_3 = d_minus * pqr_3
        else:
            #simulate wrong solution
            if np.random.rand(1) < wrong_solution_prob:
                xyz_3 = d_plus * pqr_3
            xyz_3 = d_minus * pqr_3

        xyz = np.append(xyz_1_2, xyz_3[:,np.newaxis], axis=1)

        return xyz

    def _orientation(self, xyz): #adapted from BlueSwarm code
        """Calculates the orientation angle phi using the xyz coordinates

        Args:
        xyz (float array 3x3): xyz coordinates of the three blobs (sorted)

        Returns:
        float: angle phi
        """

        phi = np.arctan2(xyz[1,2]-xyz[1,0], xyz[0,2]-xyz[0,0])

        return phi
    
    def remove_reflections(self, blobs, max_blobs):
        """Discards LED blob centroids that are considered reflections at the water surface. Reflections tend to appear higher up in the image than real centroids, i.e., they have lower m-coordinates. If the number of identified blobs is greater than the maximum number of expected blobs, the maximum number of expected blobs with the highest m-coodinates will be kept.
        """
        # print("in fish_LF/remove reflection")
        # print("before removing")
        # print(blobs)

        if np.shape(blobs)[1] > max_blobs:
            blob_ind = np.argsort(blobs[-1, :])[-max_blobs:] # note that r value is used not (different from exp fishfood lib, which uses mn coordinates with blobs[0, :] )
            blobs = blobs[:, blob_ind]

        # print("after removing")
        # print(blobs)

        return blobs
      
    def move(self, robots, rel_pos, dist, leds, abs_leds, duration):
        """Decision-making based on neighboring robots and corresponding move
        """
        # if not robots: # no other robots, continue with ctrl from last step
        #     target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)
        #     return (target_pos, self_vel)

        # Define your move here
        

        # Specify a distance range within which followers cease their actuation.
        safe_distance = self.dynamics.l_robot * 1000 * 3 # mm
        approach_distance = self.dynamics.l_robot * 1000 * 10


        # print("in move, leds\n",leds)
        # print("in move, abs_leds\n",abs_leds)
        # print("in move, rel_pos\n",rel_pos)
        # print("in move, self.id", self.id)

        if self.id == 0: # leader
            # print("************at leader************")
            magnitude = 0.2

            # self.stop()
            # self.forward(magnitude)

            self.spin( 0.1, 0.08, True) # caudal, pect, cw
            self.depth_ctrl_psensor(250,1) # target depth, dorsal freq

        elif leds.size != 0: # follower
            # print("************at follower 1************")

            # remove refection. Input leds (relative position in global frame)
            leds = leds[:3,:3] # leader's led
            # leds = self.remove_reflections(l_leds, 3) 
            leds = self.parsing(leds)  # output leds in qpr in robot's frame
            # leds = self.remove_reflections(leds, 3)
            duplet = self._pqr_to_xyz(leds)  # xyz of led_1 and led_2
            b3_pqr = leds[:,-1]
            triplet = self._pqr_3_to_xyz(duplet, b3_pqr)     
            orientation = self._orientation(triplet)    
            heading_vector = triplet[:,2] - triplet[:,0]
            # print(" heading_vector from triplet,", heading_vector)
            # print(np.linalg.norm(heading_vector))
            # input()
            
            # calculate the relative position of the leader from its LEDs (centroid is the same location as the 1st LED)
            r_move_g = triplet[:,0].transpose()
            
            # calculate the distance with respect to the leader
            rel_dist = np.linalg.norm(r_move_g[0:2])

            new_pos = self.translate( r_move_g, heading_vector,  self.id*60 , safe_distance)  #   pos, vector, keep_angle,distance)
            magnitude = np.tanh(np.linalg.norm(new_pos)/600);
            self.home(new_pos, magnitude)
            self.depth_ctrl_vision(r_move_g) 
        else:
            # print(self.id)
            ## turn clockwise if true, counter-clockwise if false?

            # self.spin(0.1, 0.1, True) # caudal, pect, cw
            self.stop() # you are dead to me



        self.dynamics.update_ctrl(self.dorsal, self.caudal, self.pect_r, self.pect_l)

        target_pos, self_vel = self.dynamics.simulate_move(self.id, duration)

        return (target_pos, self_vel)
