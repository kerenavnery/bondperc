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
    lat.p = 0.0
    lat.randomize()
    ###
    lat.percolate()
    lat.plot()
    lat.analyze()
    print(len(lat.percolators))
    dec_lat = lat.double_decimate()
    ###
    dec_lat.percolate()
    dec_lat.plot()
    dec_lat.analyze()
    print(len(dec_lat.percolators))
    
    
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
            lat.randomize()
            lat.percolate()
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
        self.p = p
        self.clusters = np.zeros((N, N), int)
        self.numclusters = 0
        self.percolators = []
        self.rightbonds = np.zeros((N, N), int)
        self.downbonds = np.zeros((N, N), int)
        
    def randomize(self):
        # Generate edges with probability p for a closed edge
        rightbonds = np.random.rand(N, N) < self.p
        downbonds = np.random.rand(N, N) < self.p
        self.rightbonds = rightbonds
        self.downbonds = downbonds
        
        
    def double_decimate(self):
        N = self.N
        if N%2 != 0:
            print("Can't divide {} by 2!".format(N))
        dec_lat = lattice(int(N/2), self.p)
        
        def is_node_on_lattice(node):
            return ((node[0] >= 0) and (node[1] >= 0) and (node[0] < N) and (node[1] < N))
        
        def is_nn_bond(node1, node2):
            if ((abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])) != 1):
                print("Error tried to find non nearest-neighbor bond")
                return False
            else:
                # node2 is above node1
                if (node1[0] - node2[0]) == 1:
                    return self.downbonds[node2[0], node2[1]]
                # node1 is above node2
                if (node1[0] - node2[0]) == -1:
                    return self.downbonds[node1[0], node1[1]]
                # node2 is left of node1
                if (node1[1] - node2[1]) == 1:
                    return self.rightbonds[node2[0], node2[1]]
                # node1 is left of node2
                if (node1[1] - node2[1]) == -1:
                    return self.rightbonds[node1[0], node1[1]]
        
        for row in range(0, self.N, 2):
            for col in range(0, self.N, 2):
                print("row = {} col = {}".format(row, col))
                
                # Current node
                node = (row, col) 
                
                # Neighbor nodes
                up      = (row, col - 1)
                right   = (row + 1, col)
                down    = (row, col + 1)
                left    = (row - 1, col)

                # Neighbors after a single decimation
                upright      = (row + 1, col - 1)
                downright    = (row + 1, col + 1)
                downleft     = (row - 1, col + 1)
                
                # Neighbors after two decimations
                rightright  = (row + 2, col)
                downdown    = (row, col + 2)
                
                # Not neighbors but relevant
                uprightright    = (row + 2, col - 1)
                downrightright  = (row + 2, col + 1)
                downdownright   = (row + 1, col + 2)
                downdownleft    = (row - 1, col + 2)
                
                ############################################
                ## I am only interested in rightright and downdown
                ####### rightright #############
                bond_node_rightright = False
                if is_node_on_lattice(rightright):
                    # Find if it is connected through one of the options
                    if is_node_on_lattice(upright):
                       bond_node_upright = (is_nn_bond(node, up) and is_nn_bond(up, upright)) or (is_nn_bond(node, right) and is_nn_bond(right, upright))  
                       bond_upright_rightright = (is_nn_bond(upright, right) and is_nn_bond(right, rightright)) or (is_nn_bond(upright, uprightright) and is_nn_bond(uprightright, rightright))  
                       bond_node_rightright = (bond_node_upright and bond_upright_rightright) or bond_node_rightright
                    if is_node_on_lattice(downright):
                       bond_node_downright = (is_nn_bond(node, down) and is_nn_bond(down, downright)) or (is_nn_bond(node, right) and is_nn_bond(right, downright))  
                       bond_downright_rightright = (is_nn_bond(downright, right) and is_nn_bond(right, rightright)) or (is_nn_bond(downright, downrightright) and is_nn_bond(downrightright, rightright))  
                       bond_node_rightright = (bond_node_downright and bond_downright_rightright) or bond_node_rightright
                    dec_lat.rightbonds[int(node[0]/2),int(node[1]/2)] = bond_node_rightright
                ####### downdown #############    
                bond_node_downdown= False
                if is_node_on_lattice(downdown):
                    # Find if it is connected through one of the options
                    if is_node_on_lattice(downleft):
                       bond_node_downleft = (is_nn_bond(node, down) and is_nn_bond(down, downleft)) or (is_nn_bond(node, left) and is_nn_bond(left, downleft))  
                       bond_downleft_downdown= (is_nn_bond(downleft, downdownleft) and is_nn_bond(downdownleft, downdown)) or (is_nn_bond(downleft, down) and is_nn_bond(down, downdown))  
                       bond_node_downdown = (bond_node_downleft and bond_downleft_downdown) or bond_node_downdown
                    if is_node_on_lattice(downright):
                       bond_node_downright = (is_nn_bond(node, down) and is_nn_bond(down,downright)) or (is_nn_bond(node, right) and is_nn_bond(right, downright))  
                       bond_downright_downdown = (is_nn_bond(downright, down) and is_nn_bond(down, downdown)) or (is_nn_bond(downright, downdownright) and is_nn_bond(downdownright, downdown))  
                       bond_node_downdown = (bond_node_downright and bond_downright_downdown) or bond_node_downdown
                    dec_lat.downbonds[int(node[0]/2),int(node[1]/2)] = bond_node_downdown
        
        return dec_lat
    
    
    
    
    def plot(self):
        nodes = []
        edges = []
        
        # Generate nodes, edges list for nx graphics
        for row in range(self.N):
            for col in range(self.N):
                #print("row = {} col = {}".format(row, col))
                # These are in the opposite format to make sure the graph is orianted correctl.
                node = (col, row) # Is actually (row, col)
                right = (col + 1, row) # Is actually (row, col +1)
                down = (col, row + 1) # Is actually (row+1, col)
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
#        print("pos = {}".format(pos))
#        print("G.nodes = {}".format(G.nodes))
#        print("nodes = {}".format(nodes))
#        print("G.edges = {}".format(G.edges))
#        print("edges = {}".format(edges))
        
        #Draw the lattice
        #nx.draw_networkx_edges(G, pos = pos)
        nx.draw_networkx(G, nodelist = nodes, pos = pos, with_labels = False, node_size = 1, edgelist = edges)
        
        #Plot it on the screen
        plt.axis('off')
        plt.show()

    def percolate(self):
        N = self.N
        self.clusters[:,:] = 0
        clusters = self.clusters
        clusteruid = int(0)
        self.uids = []
        uids = self.uids

        rightbonds = self.rightbonds
        downbonds = self.downbonds

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
                