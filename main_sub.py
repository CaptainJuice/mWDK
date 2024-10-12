import numpy as np
import sklearn.cluster as sc
import math
from scipy.sparse import csc_matrix
import clustering_methods as cmd

from utils import load_data
import clustering_metric as cm
from utils import WL,WL_noconcate,pplot2,sub_wl,IK_fm_dot
import warnings
warnings.filterwarnings('ignore')
from subgraph_alignment import subgraph_embeddings1,subgraph_embeddings2

def sub_gao(node_features, adj_mat, hop, h):
    adj_mat = csc_matrix(adj_mat)


    embedding = subgraph_embeddings1(node_features, adj_mat, h)

    return embedding
if __name__ == '__main__':
    ###################################### parm ########################################
    emb_type = {"wl_noconcate": 1,
                "ikwl_noconcate": 0}
    path1 = 'E:/Graph Clustering/dataset/real_world data/'
    path2 = 'E:/Graph Clustering/dataset/artificial data/'
    datasets2 = ['Graph_1', 'Graph_2', 'Graph_3', 'Graph_4', 'Graph_5', 'Graph_6']
    datasets1 = ['Graph_6',]


    rep = 10
    # gamma_li = [0.0001, 0.001, 0.01, 0.1, 0.5, ]
    gamma_li = [0]
    path = path2
    ####################################################################################

    if emb_type['wl_noconcate'] == 1:
        psili = [28]
        for dataset in datasets1:
            adj_mat, node_features, true_labels = load_data(path2, dataset)
            num_of_class = np.unique(true_labels).shape[0]
            acc_li,nmi_li,f1_li=[],[],[]
            for psi in psili:
                features_IK = IK_fm_dot(node_features, 300, psi)
                # features_IK =node_features
                for hop in range(7, 16):
                    print("==================================== hop: {}=================================".format(hop))
                    best_acc, best_nmi, best_f1, best_h, best_gamma = -1, -1, -1, -1, -1

                    res = np.zeros(shape=(adj_mat.shape[0], adj_mat.shape[1]))
                    for i in range(hop + 1):
                        s = np.linalg.matrix_power(adj_mat, i)
                        res += s
                    adj_mat2 = res
                    for h in range(81, 101, 2):
                        embedding = sub_wl(features_IK, adj_mat, adj_mat2, h)

                        acc,nmi,f1,para,best_labels = cmd.sc_gaussian(embedding,1,num_of_class,true_labels)
                        print('@psi={} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(psi,h,para,acc,nmi,f1))

