import numpy as np
from matplotlib.pyplot import subplots,close
from matplotlib import cm
import matplotlib
import threading

def test():
    def worker():
        cmap = []
        ncolours = 128
        for n in range(ncolours):
            cmap.append(tuple((np.random.rand(3)*256).astype(int)))
        (rows, cols) = imagedata.shape
        bitmap = np.zeros((rows, cols, 3))

        fig,ax = subplots(1,1)
        fig.show()
        img = ax.imshow(imagedata)
        img.set_interpolation('nearest')
        fig.show()
        fig.canvas.draw()

        while (not stopdrawthread.is_set()):
            if newimagedataevent.wait(1.0):
                with imagedatalock:
                 for index, value in np.ndenumerate(imagedata):
                  bitmap[index[0],index[1],:] = cmap[value % ncolours]
                img.set_data(bitmap)
                fig.canvas.draw()
                newimagedataevent.clear()
        close(fig)
        return


    pList = np.arange(10,40) / 50.0
    N = 32
    trials = 10

    imagedatalock = threading.Lock()
    newimagedataevent = threading.Event()
    stopdrawthread = threading.Event()

    a = lattice(N)
    a.generate()
    imagedata = np.empty_like(a.clusters)
    drawthread_stop = threading.Event()
    drawthread = threading.Thread(target=worker, args=())
    drawthread.start()

    results = []

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

    drawthread_stop.set()
    print "joining other threads"
    drawthread.join()
    return (pList, results)


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
        clustercount = int(0)
        right = np.random.rand(N, N) < self.p
        down = np.random.rand(N, N) < self.p

        for index, thiscluster in np.ndenumerate(self.clusters):
            if thiscluster == 0:
                clustercount += 1
                thiscluster = clustercount
                self.clusters[index] = thiscluster
            if index[0] < N - 1 and down[index]:
                self.clusters[index[0] + 1, index[1]] = thiscluster
            if index[1] < N - 1 and right[index]:
                self.clusters[index[0], index[1] + 1] = thiscluster
        self.numclusters = clustercount
        self.analyze()


    def analyze(self):
        self.sizes, null = np.histogram(self.clusters, 
                                        bins=range(self.numclusters))
        north = self.clusters[0, :]
        south = self.clusters[self.N - 1, :]
        west = self.clusters[:, 0]
        east = self.clusters[:, self.N - 1]
        self.percolators = []
        for cluster in range(self.numclusters):
            if ((cluster in north and cluster in south)
                or (cluster in west and cluster in east)):
                self.percolators.append(cluster)