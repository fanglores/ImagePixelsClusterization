from sklearn.cluster import MeanShift
from PIL import Image
import numpy as np

if __name__ == '__main__':
    image = Image.open('C:\\Users\\Katsat\\Downloads\\Plastic_Hook.jpg')

    data = np.asarray(image, dtype=int)
    dataFlatten = np.reshape(data, (-1, 3))   # data[i][j] == dataFlatten[128 * i + j])

    image.close()
    del image, data

    clustering = MeanShift(bandwidth=30, cluster_all=True, n_jobs=8).fit(dataFlatten)
    print(clustering.bandwidth)
    '''
    bandwidth is radius, should find most appropriate
    automatic bandwidth is 69.5
    '''
    print(clustering.cluster_centers_.size//3)
    print(clustering.cluster_centers_)
