import numpy as np
from sklearn.manifold import TSNE
import networkx as nx
import clustering_methods as cmd
from utils import GSNN, load_data,WL_test, WL_noconcate, IGK_WL_noconcate,IK_fm_dot,WL_noconcate_gcn,pplot2,WL_noc_sum,GSNN_try,sub_wl
import warnings
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

import scipy.io as sio
from sklearn import preprocessing
def get_neighbors_single(g, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
    return output
def get_neighbors_all(G,node,depth=1):
    temp=[node]
    single_neighbors= get_neighbors_single(G, node, 3)
    for i in range(depth):
        temp.extend(single_neighbors[i+1])
    return np.array(temp)

def plot_graph(ind,emb,h):
    fig = plt.figure(figsize=(22, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    tsne = TSNE(n_components=2, perplexity=50, n_iter=1000)
    emb_tsne = tsne.fit_transform(emb)
    # emb_tsne =emb

    i, j = np.ix_(ind, ind)
    label_neighbors = true_labels[ind]
    adj_neighbors = adj_mat[i, j]

    g = nx.from_numpy_matrix(adj_neighbors)
    mapping = {}
    pos={}
    for i, j in enumerate(ind):
        mapping[i] = j
        pos[j] = emb_tsne[j]
    e = emb_tsne[ind]
    ax1.scatter(e[:, 0], e[:, 1], s=9, cmap="rainbow")
    for i in range(len(e)):
        ax1.text(e[:, 0][i], e[:, 1][i], ind[i],
                fontsize=8, color="red", style="italic", weight="light",
                verticalalignment='center', horizontalalignment='right', rotation=0)  #
    ax1.set_title("@h={} Position of Neighbors of Node-{}".format(h,ind[0]))

    g = nx.relabel_nodes(g, mapping)

    nx.draw(g, pos=pos, with_labels=True, node_size=21, node_color=label_neighbors, font_size=18)
    ax2.set_title("@h={} Graph of Neighbors of Node-{}".format(h,ind[0]))
    plt.show()
path1 = 'E:/Graph Clustering/dataset/real_world data/'
dataset="cora"
# path1 = 'E:/Graph Clustering/dataset/artificial data/'
# dataset="Graph_9"
adj_mat, node_features, true_labels = load_data(path1, dataset)
num_of_class = np.unique(true_labels).shape[0]
deg = np.sum(adj_mat,axis=0)
post = np.where(deg>2)[0]
print(len(post),true_labels.shape[0]-len(post))
i, j = np.ix_(post, post)
true_labels= true_labels[post]
adj_mat = adj_mat[i, j]
node_features =node_features[post]
embedding = node_features


# tar = [i for i, x in enumerate(true_labels) if x == 3]

target = np.argmax(deg)
target = 1

G = nx.from_numpy_matrix(adj_mat)

list_neighbors = get_neighbors_all(G,target,depth=3)[:]
num_of_neighbors = len(list_neighbors)
plot_graph(list_neighbors,embedding,0)

# sub_graphs = [nx.subgraph(G,neighbors) for neighbors in neighbors_list]
id_nearest = list_neighbors
labels_true_nearest = true_labels[list_neighbors]
acc, nmi, f1, para, predict_labels = cmd.sc_gaussian(embedding, 1, num_of_class, true_labels)
labels_predict_nearest = predict_labels[id_nearest]
labels = true_labels
list_h = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
list_h = [1,5,9,17,20,]
for h in list_h:
    new_embedding = WL_noconcate(embedding, adj_mat, h)
    new_embedding = preprocessing.normalize(new_embedding, norm='l2')

    acc, nmi, f1, para, predict_labels = cmd.sc_linear(new_embedding, 1, num_of_class, true_labels)
    print('@{} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))

    dis = new_embedding.dot(new_embedding.T)
    dis_tar = dis[target]
    ind = dis_tar.argsort()[::-1][:num_of_neighbors]

    id_nearest =np.row_stack((id_nearest, ind))
    labels_true_nearest = np.row_stack((labels_true_nearest,true_labels[ind]))
    labels_predict_nearest = np.row_stack((labels_predict_nearest,predict_labels[ind]))
    plot_graph(ind,new_embedding,h)

print(0)
