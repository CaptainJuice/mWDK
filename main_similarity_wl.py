import argparse
import numpy as np
import sklearn.cluster as sc
import math
from utils import load_data
import clustering_metric as cm
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from utils import WL,WL_noconcate,WL_noconcate_gcn
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()




def similarity_wl(node_features, adj_mat, h):

    dist = pdist(node_features, 'euclid') #cosine
    D = squareform(dist)
    D = D/np.max(D)
    for i in range(len(D)):
        D[i] = D[i]/np.sum(D[i])
    D = 1 - D
    adj_mat = np.multiply(D,adj_mat)
    embedding = WL_noconcate(node_features,adj_mat,h)
    return embedding


if __name__ == '__main__':
    list_acc, list_nmi, list_f1 = {}, {}, {}
    rep = 10
    emb_type = 'wl'
    path = 'E:/Graph Clustering/dataset/real_world data/'
    hop = 1
    datasets = ['Graph_1', 'Graph_2', 'Graph_3', 'Graph_4', 'Graph_5', 'Graph_6', ]
    path = 'E:/Graph Clustering/dataset/artificial data/'
    for dataset in datasets:

        adj_mat, node_features, true_labels = load_data(path, dataset)
        num_of_class = np.unique(true_labels).shape[0]
        if emb_type == 'wl':
            for h in range(1, 60):
                local_nmi, local_f1, local_acc = [], [], []
                embedding = similarity_wl(node_features, adj_mat, h)
                for r in range(rep):



                    model = sc.KMeans(n_clusters=num_of_class)
                    model = model.fit(embedding)
                    predict_labels = model.predict(embedding)
                    translate, new_predict_labels = cm.translate(true_labels, predict_labels)
                    acc, nmi, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, new_predict_labels)
                    local_acc.append(acc)
                    local_nmi.append(nmi)
                    local_f1.append(f1_macro)

                list_nmi[h] = np.mean(local_nmi)
                list_f1[h] = np.mean(local_f1)
                list_acc[h] = np.mean(local_acc)
                print('@h={}: ACC:{:.6f}(std:{:.6f})  NMI:{:.6f}(std:{:.6f})  f1_macro:{:.6f}(std:{:.6f})'.format(h, np.mean(local_acc), np.std(local_acc), np.mean(local_nmi), np.std(local_nmi), np.mean(local_f1),np.std(local_f1)))
                if np.mean(local_nmi)==1:
                    break
            print("@BEST h={} ACC:{:.6f} NMI:{:.6f}  f1_macro:{:.6f}".format(max(list_nmi, key = list_nmi.get),max(list_acc.values()),max(list_nmi.values()),max(list_f1.values())))

