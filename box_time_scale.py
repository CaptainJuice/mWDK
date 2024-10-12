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


warnings.filterwarnings('ignore')

def to_onehot(prelabel):
    k = len(np.unique(prelabel))
    label = np.zeros([prelabel.shape[0], k])
    label[range(prelabel.shape[0]), prelabel] = 1
    label = label.T
    return label
def square_dist(prelabel, feature):
    if sp.issparse(feature):
        feature = feature.todense()
    feature = np.array(feature)


    onehot = to_onehot(prelabel)

    m, n = onehot.shape
    count = onehot.sum(1).reshape(m, 1)
    count[count==0] = 1

    mean = onehot.dot(feature)/count
    a2 = (onehot.dot(feature*feature)/count).sum(1)
    pdist2 = np.array(a2 + a2.T - 2*mean.dot(mean.T))

    intra_dist = pdist2.trace()
    inter_dist = pdist2.sum() - intra_dist
    intra_dist /= m
    inter_dist /= m * (m - 1)
    return intra_dist




if __name__ == '__main__':
    ###################################### parm ########################################
    emb_type = {"wl_noconcate":1,
              "ikwl_noconcate":13,
              "new_ikwl_noconcate":1
               }
    path1 = 'E:/Graph Clustering/dataset/real_world data/'
    datasets1 = ['cora','citeseer','pubmed','amap',]
    datasets2 = ['blogcatalog','flickr','wiki','dblp','acm']
    path1 = 'E:/Graph Clustering/dataset/artificial data/'
    # datasets = ['dblp']
    datasets = ['Graph_14']



    rep = 1
    ####################################################################################


    if emb_type['wl_noconcate'] == 1:#312
        for dataset in datasets:

            adj_mat, node_features, true_labels = load_data(path1, dataset)
            num_of_class = np.unique(true_labels).shape[0]
            acc_li,nmi_li,f1_li = [],[],[]
            for r in range(rep):
                best_acc, best_nmi, best_f1, best_h= -1, -1, -1, -1
                embedding = node_features.copy()
                emb = node_features.copy()
                time_start = time.perf_counter()

                new_adj = create_adj_avg(adj_mat)
                np.fill_diagonal(new_adj, 0.5)
                for h in range(20):
                    embedding = WL_noconcate_fast(embedding,new_adj)
                    embedding = preprocessing.normalize(embedding, norm='l2')
                acc,nmi,f1,para,predict_labels = cmd.sc_linear(embedding,1,num_of_class,true_labels)
                time_end = time.perf_counter()  # 记录结束时间
                time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
                print(time_sum)


    if emb_type['new_ikwl_noconcate'] == 1:
        # psili =[64,64,64,64,64,64,64,64,64,64,64,64,7,7,7,7,7,7]
        psili = [32,]
        for dataset in datasets:
            adj_mat, node_features, true_labels = load_data(path1, dataset)

            num_of_class = np.unique(true_labels).shape[0]
            acc_li,nmi_li,f1_li=[],[],[]
            for r in range(rep):
                best_acc, best_nmi, best_f1, best_h, best_psi = -1, -1, -1, -1, -1
                for psi in psili:
                    embedding = node_features.copy()
                    time_start = time.perf_counter()
                    new_adj = create_adj_avg(adj_mat)
                    for h in range(20):

                        embedding = IK_fm_dot(embedding, psi, t=200)
                        embedding = WL_noconcate_fast(embedding,new_adj)
                        embedding = preprocessing.normalize(embedding, norm='l2')
                    acc,nmi,f1,para,predict_labels = cmd.sc_linear(embedding,1,num_of_class,true_labels)

                    time_end = time.perf_counter()  # 记录结束时间
                    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
                    print(time_sum)


    if emb_type['ikwl_noconcate'] == 1:
        psili =[64,64,7,7,7,7,7,7,7,]
        for dataset in datasets:
            adj_mat, node_features, true_labels = load_data(path1, dataset)
            num_of_class = np.unique(true_labels).shape[0]
            acc_li,nmi_li,f1_li=[],[],[]
            for r in range(rep):
                best_acc, best_nmi, best_f1, best_h, best_psi = -1, -1, -1, -1, -1
                new_adj = create_adj_avg(adj_mat)
                for psi in psili:
                    embedding = IK_fm_dot(node_features, psi, t=200)
                    # embedding = node_features
                    for h in range(50):
                        embedding = WL_noconcate_fast(embedding, new_adj)
                        embedding = preprocessing.normalize(embedding, norm='l2')
                        acc,nmi,f1,para,predict_labels = cmd.sc_linear(embedding,1,num_of_class,true_labels)
                        tsne = TSNE(n_components=2, perplexity=105)
                        node_features_tsne = tsne.fit_transform(node_features)

                        tsne = TSNE(n_components=2, perplexity=105)
                        embedding_tsne = tsne.fit_transform(embedding)
                        pplot2(node_features_tsne, embedding_tsne, f'h={h}', true_labels, predict_labels,1000)
                        print('@{} psi={} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset,psi,h,para,acc,nmi,f1))
                        if best_nmi < nmi:
                            best_nmi = nmi
                            best_h = h
                            best_psi = psi
                        if best_f1 < f1:
                            best_f1 = f1
                        if best_acc < acc:
                            best_acc = acc




                print('@BEST(r= {} IKWL-{}) (psi={} h={}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(r+1,para,best_psi,best_h,best_acc,best_nmi,best_f1))
                acc_li.append(best_acc)
                nmi_li.append(best_nmi)
                f1_li.append(best_f1)
            print('@{} BEST(rep= {} IKWL-{}) (psi={} h={}): ACC:{:.6f}({:.6f})  NMI:{:.6f}({:.6f})  f1_macro:{:.6f}({:.6f})'.format(dataset,rep, para, best_psi, best_h, np.mean(acc_li),np.std(acc_li),np.mean(nmi_li),np.std(nmi_li), np.mean(f1_li),np.std(f1_li)))

#@pubmed psi=64 h=24(sc_linear): ACC:0.705533  NMI:0.320438  f1_macro:0.697425
