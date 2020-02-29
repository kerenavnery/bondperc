# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:43:26 2020

@author: Keren
"""
import numpy as np
import matplotlib.pyplot as plt

def question1 ():
    N = 1024
    realizations = 100
    p_steps = 0.05
    p_array, trajectories_found = trajectory_per_p(N,realizations, p_steps)
    
    plt.plot(p_array, trajectories_found, 'bo', p_array, trajectories_found, 'b')
    plt.title('P(trajectory) vs. p')
    plt.xlabel('p - closed edge probability')
    plt.ylabel('P - finding trajectory probability')
    plt.show()

def trajectory_per_p(N = 100, realizations = 50, p_steps = 0.5):
    #### Testing parameters ###
    # N^2 - number of sites in a square lattice. N is the number of sites in a row.
    # realizations - for each N, p: calculate percolations this number of times
    # Create an array of different "p" (bond probability)
    print("Calculating for N = {}, realizations = {}, p_steps = {}.".format(N, realizations, p_steps))
    pmin = 0
    pmax = 1
    temp_ans = np.linspace(pmin, pmax,1+ ((pmax - pmin)/p_steps), endpoint=True, retstep=True)
    # Check that the required p_steps was generated
    if temp_ans[1] != p_steps:
        print("Oh no! Probability array was generated not as required. s_steps required is {}, s_steps recieved is {}".format(p_steps, temp_ans(1)))
    p_array = temp_ans[0]

    ### Initialize test ###
    lat = lattice(N)
    trajectories_found = []

    ### Run  test ###
    for p in p_array:
        print()
        print("Calculating for p = {}".format(p))
        # Define a lattice with p bond probability
        lat.p = p
        number_of_percolating_realizations = 0
        for t in range(realizations):
            # Generate a random lattice
            lat.generate()
            # Find percolations for the given lattice
            lat.analyze()
            # Count if this realization contain percolations
            if len(lat.percolators) > 0:
                number_of_percolating_realizations += 1
        trajectories_found.append(number_of_percolating_realizations/realizations)
        print("Found a trajectory in {} of the cases".format(number_of_percolating_realizations/realizations))

    return (p_array, trajectories_found)


class lattice(object):
    def __init__(self, N=16, p=0.5):
        self.N = N
        self.clusters = np.zeros((N, N), int)
        self.numclusters = 0
        self.p = p
        self.percolators = []
        self.sizes = []

    def generate(self):
        N = self.N
        self.clusters[:,:] = 0
        clusters = self.clusters
        clusteruid = int(0)
        self.uids = []
        uids = self.uids
        rightbonds = np.random.rand(N, N) < self.p
        downbonds = np.random.rand(N, N) < self.p

#        for index, thiscluster in np.ndenumerate(self.clusters):
#            if thiscluster == 0:
#                clustercount += 1
#                thiscluster = clustercount
#                self.clusters[index] = thiscluster
#            if index[0] < N - 1 and down[index]:
#                self.clusters[index[0] + 1, index[1]] = thiscluster
#            if index[1] < N - 1 and right[index]:
#                self.clusters[index[0], index[1] + 1] = thiscluster
        
        for row in range(N):
            for col in range(N):
                right = (row, col + 1)
                down = (row + 1, col)
                clusterID = clusters[row, col]

                if clusterID == 0:
                    ## new cluster
                    clusteruid += 1
                    clusterID = clusteruid
                    clusters[row,col] = clusterID
                    uids.append(clusterID)

                if col < N - 1 and rightbonds[row,col]:
                    if clusters[right] == 0:
                        ## nothing to the right
                        clusters[right] = clusterID
                    elif clusterID != clusters[right]:
                        ## different cluster found to right
                        existingcluster = clusters[right]
                        clusters[clusters == clusterID] = existingcluster
                        uids.remove(clusterID)
                        clusterID = existingcluster
                if row < N - 1 and downbonds[row, col]:
                    self.clusters[down] = clusterID

        self.numclusters = len(uids)
        self.analyze()


    def analyze(self):
        self.sizes, null = np.histogram(self.clusters, 
                                        bins=range(self.numclusters))
        north = self.clusters[0, :]
        south = self.clusters[self.N - 1, :]
        west = self.clusters[:, 0]
        east = self.clusters[:, self.N - 1]
        self.percolators = []
        for cluster in self.uids:
            if ((cluster in north and cluster in south)
                or (cluster in west and cluster in east)):
                self.percolators.append(cluster)
                