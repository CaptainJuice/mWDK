import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from utils import GSNN, load_data,WL, WL_noconcate_one,WL_noconcate, IGK_WL_noconcate,IK_inne_fm,IK_fm_dot,WL_noconcate_gcn,pplot2,GSNN_try,WL_gao
from utils import create_adj_avg,WL_noconcate_fast,create_adj_avg_gcn
from sklearn.manifold import TSNE

def gen_random_3d_graph(n_nodes, radius):
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for i in range(n_nodes)}
    graph = nx.random_geometric_graph(n_nodes, radius, pos=pos)
    return graph

def mygraph():
    path1 = 'E:/Graph Clustering/dataset/artificial data/'
    dataset = 'Graph_21'
    adj_mat, node_features, true_labels = load_data(path1, dataset)
    tsne = TSNE(n_components=3, perplexity=31)
    node_features_tsne = tsne.fit_transform(node_features)
    pos = {i: (node_features_tsne[:, 0][i], node_features_tsne[:, 1][i],node_features_tsne[:, 2][i]) for i in range(len(true_labels))}
    G = nx.from_numpy_matrix(adj_mat)

    return G,pos



def plot_3d_network(graph, angle,pos):
    # pos = nx.get_node_attributes(graph, 'pos')
    with plt.style.context("bmh"):
        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            ax.scatter(xi, yi, zi, edgecolor='b', alpha=0.9)
            for i, j in enumerate(graph.edges()):
                x = np.array((pos[j[0]][0], pos[j[1]][0]))
                y = np.array((pos[j[0]][1], pos[j[1]][1]))
                z = np.array((pos[j[0]][2], pos[j[1]][2]))
                ax.plot(x, y, z, c='black', alpha=0.9)
    ax.view_init(30, angle)
    pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))
    plt.show()


if __name__ == '__main__':
    # graph01 = gen_random_3d_graph(15, 0.6)

    graph01,pos = mygraph()
    plot_3d_network(graph01, 0,pos)
