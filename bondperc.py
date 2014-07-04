import numpy as np
from matplotlib.pyplot import subplots,close
from matplotlib import cm, animation, pyplot
import matplotlib
import threading

def test():
    pList = np.arange(0.4,0.6,0.02)
    N = 32
    trials = 100

    imagedatalock = threading.Lock()
    newimagedataevent = threading.Event()

    a = lattice(N)
    a.generate()
    imagedata = np.empty_like(a.clusters)
    results = []
    times = []

    def worker():
        for p in pList:
            a.p = p
            percolating = 0
            for t in range(trials):
                a.generate()
                a.analyze()

                if imagedatalock.acquire(False):
                    imagedata[:] = a.clusters
                    imagedatalock.release()
                    newimagedataevent.set()

                if len(a.percolators) > 0: percolating += 1
            results.append(percolating)
        return results

    cmap = []
    ncolours = 32
    for n in range(ncolours):
        cmap.append(tuple((np.random.rand(3)*256).astype(int)))
    (rows, cols) = imagedata.shape
    bitmap = np.zeros((rows, cols, 3))
    fig,ax = subplots(1,1)
    img = ax.imshow(imagedata)
    img.set_interpolation('nearest')
    fig.canvas.draw()

    workerthread = threading.Thread(target=worker)
    workerthread.start()

    def figupdate(*args):
        if newimagedataevent.wait(1.0):
            with imagedatalock:
                for index, value in np.ndenumerate(imagedata):
                    bitmap[index[0],index[1],:] = cmap[value % ncolours]
                img.set_array(bitmap)
        return img

    ani = animation.FuncAnimation(fig, figupdate)
    fig.show()   

    while workerthread.is_alive():
        fig.canvas.flush_events()

    workerthread.join()
    return (pList, results, times)

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