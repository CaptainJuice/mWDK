import numpy as np
from utils import IK_fm_dot
from utils import load_data,GSNN,GSNN_try,pplot2,create_adj_avg_gcn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import clustering_methods as cmd
import scipy
import warnings
warnings.filterwarnings('ignore')
import time

import ogb
from ogb.nodeproppred import NodePropPredDataset

if __name__ == '__main__':




    path1 = 'E:/Graph Clustering/dataset/real_world data/'
    # path1 = 'E:/Graph Clustering/dataset/artificial data/'
    datasets = ['cora', ]
    # datasets =['ENodes_EDegrees_Hard']

    rep = 2


    for dataset in datasets:
        adj_mat, node_features, true_labels = load_data(path1, dataset)
        num_of_class = len(set(true_labels))
        list_nmi, list_f1, list_acc = [], [], []
        time_start = time.perf_counter()
        for h in range(1,10):
            embedding = GSNN(node_features, adj_mat=adj_mat, h=h)

            # acc, nmi, f1, para,best_labels = cmd.sc_linear(embedding, 1, num_of_class, true_labels)

            # list_nmi.append(nmi)
            # list_f1.append(f1)
            # list_acc.append(acc)
            # print('@{} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))
            # tsne = TSNE(n_components=2, perplexity=45)
            # node_features_tsne = tsne.fit_transform(node_features)
            #
            # tsne = TSNE(n_components=2, perplexity=45)
            # embedding_tsne = tsne.fit_transform(embedding)
            # pplot2(node_features_tsne, embedding_tsne, f'h={h}', true_labels, predict_labels, p=100)
        print('@GraphSNN@best: ACC:{:.6f}(std:{:.6f})  NMI:{:.6f}(std:{:.6f})  f1_macro:{:.6f}(std:{:.6f})'.format(np.max(list_acc), np.std(list_acc),np.max(list_nmi), np.std(list_nmi), np.max(list_f1), np.std(list_f1)))
        time_end = time.perf_counter()
        print(time_start-time_end)

