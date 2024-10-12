import copy
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score,normalized_mutual_info_score,accuracy_score
from munkres import Munkres
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
import clustering_methods as cmd
from utils import GSNN, load_data,WL, WL_noconcate_one,WL_noconcate, IGK_WL_noconcate,IK_inne_fm,IK_fm_dot,WL_noconcate_gcn,pplot2,GSNN_try,WL_gao
from utils import create_adj_avg,WL_noconcate_fast,create_adj_avg_gcn
import warnings
import scipy.io as sio
from sklearn import preprocessing
from Lambda_feature import lambda_feature_continous
from pyvis.network import Network
from IPython.core.display import display, HTML
import pandas as pd

def plot(X, fig, col, size, true_labels):
    ax = fig.add_subplot(1, 1, 1)
    for i, point in enumerate(X):
        ax.scatter(point[0], point[1], s=size, c=col[true_labels[i]])


def plotClusters(tqdm, hidden_emb, true_labels,dataset,name):
    tqdm.write('Start plotting using TSNE...')
    # Doing dimensionality reduction for plotting
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(hidden_emb)
    # Plot figure
    fig = plt.figure()
    plot(X_tsne, fig, ['red', 'green', 'blue', 'brown', 'purple', 'yellow', 'pink', 'orange'], 4, true_labels)
    # fig.show()
    fig.savefig("./visualizations/{}/{}.png".format(dataset,name))
    tqdm.write("Finished plotting")


def visualize(dataset):


    adj_mat, node_features, true_labels = load_data(path1, dataset)
    num_of_class = np.unique(true_labels).shape[0]
    np.where(adj_mat != 0, adj_mat, 1)
    np.fill_diagonal(adj_mat, 0)
    acc_li, nmi_li, f1_li = [], [], []

    best_acc, best_nmi, best_f1, best_h = -1, -1, -1, -1

    emb = node_features.copy()
    embedding = node_features.copy()

    new_adj = create_adj_avg(adj_mat)
    for h in range(31):
        if h > 0:
            embedding = WL_noconcate_fast(embedding, new_adj)
            embedding = preprocessing.normalize(embedding, norm='l2', axis=0, )
            tsne = TSNE(n_components=2, perplexity=31)
            node_features_tsne = tsne.fit_transform(node_features)

            tsne = TSNE(n_components=2, perplexity=31)
            embedding_tsne = tsne.fit_transform(embedding)
            acc, nmi, f1, para, predict_labels = cmd.sc_linear(embedding, 1, num_of_class, true_labels)

            pplot2(node_features_tsne, embedding_tsne, f'h={h}', true_labels, predict_labels, p=800)
            print('@{} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))


            # pos = copy.deepcopy(embedding_tsne)
            # plt.figure(dpi=500)
            #
            #
            # nx.draw_networkx_nodes(G, pos, node_size=3, node_color=true_labels)  # 画节点
            # nx.draw_networkx_edges(G, pos, alpha=0.6, width=0.5)  # 画边
            # plt.show()

def visualize2(dataset):
    adj_mat, node_features, true_labels = load_data(path1, dataset)
    num_of_class = np.unique(true_labels).shape[0]
    np.where(adj_mat != 0, adj_mat, 1)
    np.fill_diagonal(adj_mat, 0)
    embedding = node_features.copy()
    adj = sp.coo_matrix(adj_mat)
    indices = np.vstack((adj.row, adj.col))
    df = pd.DataFrame(indices)
    df =df.T
    df['weight']=1
    df.columns = ['source', 'target','value']
    df.to_csv('{}.csv'.format(dataset),index=False)
    print("ok")





    new_adj = create_adj_avg(adj_mat)
    for h in range(31):
        if h > 0:
            embedding = WL_noconcate_fast(embedding, new_adj)
            embedding = preprocessing.normalize(embedding, norm='l2', axis=0, )
            tsne = TSNE(n_components=2, perplexity=31)
            node_features_tsne = tsne.fit_transform(node_features)

            tsne = TSNE(n_components=2, perplexity=31)
            embedding_tsne = tsne.fit_transform(embedding)
            acc, nmi, f1, para, predict_labels = cmd.sc_linear(embedding, 1, num_of_class, true_labels)

            print('@{} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))

            G = nx.from_numpy_matrix(adj_mat)
            net = Network(notebook=True,cdn_resources='remote')
            net.from_nx(G)
            net.show("{}.html".format(dataset))
            display("{}.html".format(dataset))


if __name__ == '__main__':
    emb_type = {"wl_noconcate": 1,
                "ikwl_noconcate": 12,
                "new_ikwl_noconcate": 12
                }
    path1 = 'E:/Graph Clustering/dataset/real_world data/'
    datasets1 = ['cora', 'citeseer', 'pubmed', 'amap', ]
    datasets2 = ['blogcatalog', 'flickr', 'wiki', 'dblp', 'acm']
    # path1 = 'E:/Graph Clustering/dataset/artificial data/'
    dataset='imbalanced'
    dataset='cora'


    visualize2(dataset)
