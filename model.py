from sklearn.cluster import MeanShift
from PIL import Image
import numpy as np
import time
import math

class Model:
    def __init__(self):
        # Stores
        self._bandwidth1 = None
        self._bandwidth2 = None
        self._clusterArray1 = []
        self._clusterArray2 = []
        self._clustersMap = []
        print('Model created')

    def __del__(self):
        print('Model destroyed')

    def SetBandwidth(self, bandwidth1 = None, bandwidth2 = None):
        if bandwidth1 is not None:
            self._bandwidth1 = bandwidth1

        if bandwidth2 is not None:
            self._bandwidth2 = bandwidth2

    def __Distance(self, pixel1, pixel2):
        return math.sqrt((pixel1[0] - pixel2[0])**2 + (pixel1[1] - pixel2[1])**2 + (pixel1[2] - pixel2[2])**2)

    def __GetPixelNumForCluster(self, clusterCenter, bandwidth, array):
        for i in range(0, len(array)):
            if self.__Distance(clusterCenter, array[i]) <= bandwidth:
                return i

    # Probably there is a better way to get cluster by value using library?
    def __GetClusterNumForPixel(self, pixel, bandwidth, clusterArray):
        for i in range(0, len(clusterArray)):
            if self.__Distance(pixel, clusterArray[i]) <= bandwidth:
                return i

    def __MapClusters(self, pixelsArray1, pixelsArray2):
        # dummy mode!
        for numCluster1 in range(0, len(self._clusterArray1)):
            pixelNum = self.__GetPixelNumForCluster(self._clusterArray1[numCluster1], self._bandwidth1, pixelsArray1)
            numCluster2 = self.__GetClusterNumForPixel(pixelsArray2[pixelNum], self._bandwidth2, self._clusterArray2)

            self._clustersMap.append([numCluster1, numCluster2])

    def Learn(self, pathToPhoto1, pathToPhoto2, bandwidth1=30, bandwidth2=30):
        self._bandwidth1 = bandwidth1
        self._bandwidth2 = bandwidth2

        print('Opening and parsing images')
        # Get image. Use only jpg type (png has 4 values, 4th for storing alpha channel value)
        image1 = Image.open(pathToPhoto1)
        image2 = Image.open(pathToPhoto2)

        # Parse image into integer array and make it 1d array of pixels
        data1 = np.asarray(image1, dtype=int)
        pixelsArray1 = np.reshape(data1, (-1, 3))  # data[i][j] == dataFlatten[128 * i + j])

        data2 = np.asarray(image2, dtype=int)
        pixelsArray2 = np.reshape(data2, (-1, 3))

        # Get rid of unwanted further variables
        image1.close()
        image2.close()
        del image1, image2
        del data1, data2

        print(f'Starting learning with {self._bandwidth1} and {self._bandwidth2} bandwidths respectively')

        # Perform clusterization using MeanShift method.
        # It allows to generate clusters, defining only the distance between objects (bandwidth)
        # Cluster_all means that the orphan objects in the end will be assigned to the closest cluster
        # N_jobs allows to utilize CPU more and greatly increases the efficiency
        startLearningTimePoint = time.time()
        clustering1 = MeanShift(bandwidth=self._bandwidth1, cluster_all=True, n_jobs=8).fit(pixelsArray1)
        clustering2 = MeanShift(bandwidth=self._bandwidth2, cluster_all=True, n_jobs=8).fit(pixelsArray2)
        endLearningTimePoint = time.time()

        '''
        bandwidth is radius, should find most appropriate
        automatic bandwidth is 69.5
        too big radius means few clusters, when too small radius means a lot of clusters
        value needed what will give around 32 clusters per any image on input?
        '''

        # Print number of found clusters and its centers
        print(f'Performed learning took {(endLearningTimePoint - startLearningTimePoint):.2f} seconds'
              f' with result of {clustering1.cluster_centers_.size // 3}'
              f' and {clustering2.cluster_centers_.size // 3} clusters accordingly')

        self._clusterArray1 = clustering1.cluster_centers_
        self._clusterArray2 = clustering2.cluster_centers_

        self.__MapClusters(pixelsArray1, pixelsArray2)
