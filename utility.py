import numpy as np
import scipy.io as scio
import os
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist


def load_graph(filename):
    dir = 'dataset'
    graph = scio.loadmat(os.path.join(dir, filename))
    attr = graph['Attributes']
    adj = graph['Network']
    label = graph['Label']
    return attr, adj, label


def adj_transform(adj, dis_matrix, band):
    data = dis_matrix.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=band).fit(data)
    weight_matrix = np.exp(kde.score_samples(data)).reshape(dis_matrix.shape[0], dis_matrix.shape[1])
    return np.multiply(adj + np.eye(dis_matrix.shape[0]), dis_matrix)

def main():
    data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    adj = np.array([0])
    adj_transform(adj, cdist(data, data))


if __name__ == '__main__':
    main()
