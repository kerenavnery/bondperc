# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:43:26 2020

@author: Keren
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def test ():
    lat = lattice(4)
    lat.p = 0.5
    lat.generate()
    lat.plot()
    lat.analyze()
    print(len(lat.percolators))
    
    
def question1 ():
    N = 10
    realizations = 10
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

def plot_lattice(lat):
    N = lat.N
    rightbonds = lat.rightbonds
    downbonds = lat.downbonds
    G = nx.grid_2d_graph(N,N)
    #We need this so that the lattice is drawn vertically to the horizon
    pos = dict( (l,l) for l in G.nodes() )
    print("pos = {}".format(pos))
    print("G.nodes = {}".format(G.nodes))
    nodes = [(0, 0), (0, 1), (1,0), (1, 1)]
    print("nodes = {}".format(nodes))
    print("G.edges = {}".format(G.edges))
    edges = [ ((0, 0), (0, 1)), ((0, 1), (1, 1))]
    print("edges = {}".format(edges))
    
    #Draw the lattice
    #nx.draw_networkx_edges(G, pos = pos)
    nx.draw_networkx(G, nodelist = nodes, pos = pos, with_labels = False, node_size = 0, edgelist = edges)
    
    #Plot it on the screen
    plt.axis('off')
    plt.show()
    
class lattice(object):
    def __init__(self, N=16, p=0.5):
        self.N = N
        self.clusters = np.zeros((N, N), int)
        self.numclusters = 0
        self.p = p
        self.percolators = []
        self.rightbonds = np.zeros((N, N), int)
        self.downbonds = np.zeros((N, N), int)
    
    def plot(self):
        nodes = []
        edges = []
        
        # Generate nodes, edges list for nx graphics
        for row in range(self.N):
            for col in range(self.N):
                print("row = {} col = {}".format(row, col))
                node = (col, row) 
                right = (col + 1, row)
                down = (col, row + 1)
                nodes.append(node)
                # THere is a bond to the right
                if col < self.N - 1 and self.rightbonds[row,col]:
                    edges.append((node, right))
                # There is a bond downwards
                if row < self.N - 1 and self.downbonds[row, col]:
                    edges.append((node, down))    
                
        
        G = nx.grid_2d_graph(self.N,self.N)
        #We need this so that the lattice is drawn vertically to the horizon
        pos = dict( (l,l) for l in G.nodes() )
        print("pos = {}".format(pos))
        print("G.nodes = {}".format(G.nodes))
        print("nodes = {}".format(nodes))
        print("G.edges = {}".format(G.edges))
        print("edges = {}".format(edges))
        
        #Draw the lattice
        #nx.draw_networkx_edges(G, pos = pos)
        nx.draw_networkx(G, nodelist = nodes, pos = pos, with_labels = False, node_size = 1, edgelist = edges)
        
        #Plot it on the screen
        plt.axis('off')
        plt.show()

    def generate(self):
        N = self.N
        self.clusters[:,:] = 0
        clusters = self.clusters
        clusteruid = int(0)
        self.uids = []
        uids = self.uids
        
        # Generate edges with probability p for a closed edge
        rightbonds = np.random.rand(N, N) < self.p
        downbonds = np.random.rand(N, N) < self.p
        self.rightbonds = rightbonds
        self.downbonds = downbonds

        # Loop over sites and check connectivity via edges to other sites
        # Start at the upper-left corner
        for row in range(N):
            for col in range(N):
                # The bonds to the right and downwards
                right = (row, col + 1)
                down = (row + 1, col)
                clusterID = clusters[row, col]

                # Is this site connected to a previously defined cluster?
                if clusterID == 0:
                    ## new cluster
                    clusteruid += 1
                    clusterID = clusteruid
                    clusters[row,col] = clusterID
                    uids.append(clusterID)

                # THere is a bond to the right
                if col < N - 1 and rightbonds[row,col]:
                    ## not an existing cluster to the right
                    if clusters[right] == 0:
                        clusters[right] = clusterID
                    ## Found an existing cluster to the right
                    elif clusterID != clusters[right]:
                        existingcluster = clusters[right]
                        # relabel the connected clusters the same
                        clusters[clusters == clusterID] = existingcluster
                        uids.remove(clusterID)
                        clusterID = existingcluster
                # There is a bond downwards
                if row < N - 1 and downbonds[row, col]:
                    ## percolate! connect the site downwards to the cluster above it.
                    self.clusters[down] = clusterID

        self.numclusters = len(uids)


    def analyze(self):
        # Find which clusters are percolating from top to bottom
        north = self.clusters[0, :]
        south = self.clusters[self.N - 1, :]
        self.percolators = []
        for cluster in self.uids:
            if (cluster in north and cluster in south):
                self.percolators.append(cluster)
                