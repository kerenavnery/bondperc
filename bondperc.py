import numpy as np

N = 8
p = 0.5

bonds = np.zeros((N, N))
clusters = np.zeros((N, N), int)

right = np.random.rand(N, N) < p
down = np.random.rand(N, N) < p

clustercount = int(0)

for index, thiscluster in np.ndenumerate(clusters):
	if thiscluster == 0:
		clustercount += 1
		thiscluster = clustercount
		clusters[index] = thiscluster
	if index[0] < N - 1 and down[index]:
			clusters[index[0] + 1, index[1]] = thiscluster
	if index[1] < N - 1 and right[index]:
		clusters[index[0], index[1] + 1] = thiscluster

