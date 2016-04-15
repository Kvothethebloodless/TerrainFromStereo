import numpy as np


# import matplotlib.pyplot as plt
# import tables as tb

def scale_array(arr, max_vl, min_vl):
    arr_min = np.min(np.array(arr))
    arr_max = np.max(np.array(arr))
    scaled_array = (arr - arr_min)
    scaled_array = scaled_array * (max_vl - min_vl)
    return scaled_array + min_vl


class PSO:
    def __init__(self, score_func, no_particles, no_dim, self_accel_coeff, global_accel_coeff, dt, one_point,
                 searchspaceboundaries):
        """no_particles, no_dim, self_accel_coeff, global_accel_coeff, dt , A total of 5 parameters
		no_particles = Number of swarm particles to create and engage
		no_dim = Dimension of the search space 	. Searchboundaries is a 2byndim array with the first component being the inner boundary values and the second
		component being the outer boundary values"""
        self.score_func = score_func
        self.no_particles = no_particles
        self.no_dim = no_dim

        self.self_accel_coeff = self_accel_coeff
        self.global_accel_coeff = global_accel_coeff

        self.dt_velocity = 1
        self.dt_pos = 1
        self.weight = .5
        self.maxvel = 10
        self.minvel = 0

        self.searchspaceboundaries = searchspaceboundaries
        self.innerboundary = self.searchspaceboundaries[0]
        self.outerboundary = self.searchspaceboundaries[1]

        self.initialize_swarm()

    def initialize_swarm(self):
        self.velocity = np.random.random((self.no_particles, self.no_dim))
        # self.current_pos = (np.random.random((self.no_particles, self.no_dim))) * 10 - (
        #     5 * np.ones((self.no_particles, self.no_dim)))
        # self.current_pos =
        list_pos = []  # An empty list to hold lists of positions of each particle in each dimension. i.e, if there are 10 dims and 50 particles, this list will
        # consist of 10 lists of 50 sub entries which specify the dimension of each particle in that dimension.

        for i in range(self.no_dim):
            # print i
            ithdim_pos = np.random.uniform(self.innerboundary[i], self.outerboundary[i], self.no_particles)
            list_pos.append(ithdim_pos)

        pos_array = np.array(list_pos)
        self.current_pos = np.reshape(np.transpose(pos_array), (self.no_particles, self.no_dim))

        self.paramspacebounds = [-10, 10]
        self.selfminlocation = self.current_pos  # np.random.random((self.no_particles,self.no_dim));
        self.selfminval = self.funcdef(self.selfminlocation, self.no_particles)
        self.globalmin = np.min(self.selfminval)
        self.globalminlocation = self.selfminlocation[np.argmin(self.selfminval)]
        self.curr_score = self.funcdef(self.current_pos, self.no_particles)
        self.centroid = np.mean(self.current_pos, axis=0)
        self.spread = 0

    def update_selfmin(self):
        update_req_array = self.curr_score < self.selfminval

        self.selfminval = np.multiply(update_req_array, self.curr_score) + np.multiply(np.invert(update_req_array),
                                                                                       self.selfminval)

        update_req_array = np.transpose(update_req_array)

        update_req_array = np.reshape(update_req_array, (self.no_particles, 1))
        # print(np.shape(update_req_array));

        # print(np.shape(self.current_pos));
        self.selfminlocation = (update_req_array * self.current_pos) + (
            np.invert(update_req_array) * self.selfminlocation)

    def update_globalmin(self):
        curr_globalmin = np.amin(self.curr_score)
        curr_globalmin_loc = self.current_pos[self.curr_score == curr_globalmin]
        curr_globalmin_loc = curr_globalmin_loc[0]
        state = curr_globalmin < self.globalmin
        state = np.squeeze(state)
        self.globalmin = np.multiply(state, curr_globalmin) + np.multiply(np.invert(state), self.globalmin)
        self.globalminlocation = (state * curr_globalmin_loc) + (np.invert(state) * self.globalminlocation)

    def funcdef(self, locs, no_particles):
        score_curr = np.empty(no_particles)

        for i in range(no_particles):
            score_curr[i] = self.score_func(locs[i])
        # print(score_curr[i])
        return score_curr

    def update_currscores(self):
        self.curr_score = self.funcdef(self.current_pos, self.no_particles)

    def update_velocities(self):
        self_accel = np.random.random() * (self.selfminlocation - self.current_pos)
        global_accel = np.random.random() * (self.globalminlocation - self.current_pos)
        accel = self.self_accel_coeff * self_accel + self.global_accel_coeff * global_accel
        self.velocity = self.weight * self.velocity + accel * self.dt_velocity
        self.velocity[self.velocity > self.maxvel] = self.maxvel
        self.velocity[self.velocity < (-self.maxvel)] = -self.maxvel

    def update_pos(self):
        self.current_pos += self.velocity * self.dt_pos
        self.current_pos = self.searchspacelimit(self.current_pos)

    def searchspacelimit(self, pos):
        for i in range(self.no_dim):
            pos[pos[:, i] > self.outerboundary[i], i] = self.outerboundary[i]
            pos[pos[:, i] < self.innerboundary[i], i] = self.innerboundary[i]
        return pos

    def calc_swarm_props(self):
        centroid = np.mean(self.current_pos, axis=0)
        spread = 0
        for i in range(self.no_particles):
            spread += np.linalg.norm(self.current_pos - centroid)
        self.centroid = centroid
        self.spread = spread
        self.best_score = np.min(self.curr_score)
        return centroid, spread

    def report(self):
        print("The centroid is at " + str(self.centroid) + "\n")
        print("The spread is " + str(self.spread) + "\n")
        print("------")
        print("The current best score  is" + str(np.min(self.curr_score)) + "\n")
        print("The global best is at " + str(self.globalminlocation) + "\n")
        print("The gloabl best score is" + str(self.globalmin) + "\n")
        # print("Ca")

        if self.no_particles < 20:
            print(
                "current score \t\t best self score\t\t current_position \t\t best self position \t\t current velocity \n")
            for i in range(self.no_particles):
                print(str(self.curr_score[i]) + '\t\t ' + str(self.selfminval[i]) + '\t\t ' + str(
                        self.current_pos[i]) + '\t\t  ' + str(self.selfminlocation[i]) + '\t\t ' + str(
                        self.velocity[i]))
            # print(self.current_pos[i])
            # print(self.velocity[i])
            # print("The current score is  "+str(self.curr_score)+"\n");
            # print("The positions are " + str(self.current_pos)+"\n");
            # print("-----");


            # print("The current velocities are "+ str(self.velocity)+"\n");
            print("#$#$#$#$#$#$ \n")
