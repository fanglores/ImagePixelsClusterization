from sklearn.cluster import MeanShift
from PIL import Image
import numpy as np

if __name__ == '__main__':
    # Get image. Use only jpg type (png has 4 values, 4th for storing alpha channel value)
    image = Image.open('C:\\Users\\Katsat\\Downloads\\Plastic_Hook.jpg')

    # Parse image into integer array and make it 1d array of pixels
    data = np.asarray(image, dtype=int)
    dataFlatten = np.reshape(data, (-1, 3))   # data[i][j] == dataFlatten[128 * i + j])

    # Get rid of unwanted further variables
    image.close()
    del image, data

    # Perform clusterization using MeanShift method.
    # It allows to generate clusters, defining only the distance between objects (bandwidth)
    # Cluster_all means that the orphan objects in the end will be assigned to the closest cluster
    # N_jobs allows to utilize CPU more and greatly increases the efficiency
    clustering = MeanShift(bandwidth=30, cluster_all=True, n_jobs=8).fit(dataFlatten)

    '''
    bandwidth is radius, should find most appropriate
    automatic bandwidth is 69.5
    too big radius means few clusters, when too small radius means a lot of clusters
    value needed what will give around 32 clusters per any image on input?
    '''

    # Print number of found clusters and its centers
    print(clustering.cluster_centers_.size//3)
    print(clustering.cluster_centers_)
