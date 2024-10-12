import copy

import sklearn.cluster as sc
import numpy as np
import clustering_metric as cm
from tqdm import tqdm
from utils import load_data,WL_noconcate_fast,create_adj_avg,IK_fm_dot
import clustering_methods as cmd
import sys
# from run import main
sys.path.append("..")
from sklearn.metrics.cluster import  normalized_mutual_info_score
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing
warnings.filterwarnings('ignore')

def heterphlic(nodes,adj_mat,true_labels):
    res = []
    cnt = 0
    for i in nodes:
        neighbors = np.where(adj_mat[i]!=0)
        neighbors_labels = true_labels[neighbors]
        diff = len(set(neighbors_labels))
        if diff >1:
            res.append(i)
            cnt+=1
    return res,cnt
NMI = lambda x, y: normalized_mutual_info_score(x, y, average_method='arithmetic')
datasets = ['cora']
path = 'E:/Graph Clustering/dataset/real_world data/'
for dataset in datasets:
    adj_mat, node_features, true_labels = load_data(path,dataset)
    print("**************************************DATASET: {} **************************************".format(dataset))


    # @h=11 acc:0.6971935007385525  nmi:0.5605251585747875  f1:0.6971112447102241
    # new_adj = create_adj_avg(adj_mat)
    # embedding = copy.deepcopy(node_features)
    # num_of_class = len(list(set(true_labels)))
    # num_of_nodes = len(true_labels)
    # for h in range(7):
    #     if h >0:
    #         myemb = copy.deepcopy(embedding)
    #         embedding = IK_fm_dot(embedding, psi=64, t=100)
    #         embedding = preprocessing.normalize(embedding, norm='l2', axis=0)
    #
    #         embedding = WL_noconcate_fast(embedding,new_adj)
    #     emb = preprocessing.normalize(embedding, norm='l2', axis=1)
    #
    #     acc, nmi, f1, para, predict_labels = cmd.sc_linear(emb, 1, num_of_class, true_labels)
    #     print("==================={}==============".format(h))
    #     if h >= 1:
    #         percent = 0.01
    #
    #         cluster = [[] for _ in range(num_of_class)] # save the index of points of each cluster
    #         features = [[] for i in range(num_of_class)]  # save the features of points of each cluster
    #         for index in range(num_of_nodes):
    #             label = predict_labels[index]
    #             cluster[label].append(index)
    #             features[label].append(embedding[index])
    #
    #
    #
    #         del_li=[]
    #         features =np.array(features)
    #         for i in range(num_of_class):
    #             inner_feature =np.array(features[i])
    #             inner_center = np.mean(inner_feature,axis=0)
    #             score = inner_feature.dot(inner_center)
    #             index_of_sort_score = np.argsort(score)
    #             index_li = [cluster[i][t]for t in index_of_sort_score]
    #             n = len(score)
    #             sort_score = np.zeros(n)
    #             for i in range(n):
    #                 sort_score[i] = score[index_of_sort_score[i]]
    #             p = int(len(index_li)*percent)
    #             del_li.extend(index_li[:p])
    #         ans = heter(del_li,adj_mat,true_labels)
    #
    #
    #         noise_rate = round(len(del_li)*100 / num_of_nodes, 2)
    #         my_dict = {}
    #         for index, value in enumerate(true_labels):
    #             my_dict[index] = value
    #
    #         for index in del_li:
    #             my_dict.pop(index)
    #
    #         new_true_labels = list(my_dict.values())
    #
    #         my_dict = {}
    #         for index, value in enumerate(predict_labels):
    #             my_dict[index] = value
    #
    #         for index in del_li:
    #             my_dict.pop(index)
    #
    #         new_predict_labels = list(my_dict.values())
    #
    #
    #
    #         nmi2 =NMI(new_true_labels,new_predict_labels)
    #         print("@h={} drops={} nmi_ori={} nmi_post={} rates={}".format(h,len(del_li),nmi,nmi2,ans/len(del_li)))
    #
    #         # adj_mat1 = np.array(adj_mat[0])
    #         # adj=[]  # save the adjacency matrix of points of each cluster
    #         # for id in range(num_of_class):
    #         #     i, j = np.ix_(cluster[id],cluster[id])
    #         #     adj.append(adj_mat1[i,j])
    #         adj = copy.deepcopy(adj_mat)
    #         feat = copy.deepcopy(myemb)
    #         true_lab = copy.deepcopy(true_labels)
    #         adj = np.delete(adj, del_li, axis=0)  # 删除第3行 axis用于控制行列
    #         adj = np.delete(adj, del_li, axis=1)
    #         true_lab=np.delete(true_lab,del_li,axis=0)
    #         feat = np.delete(feat,del_li,axis=0)# 删除第3行 axis用于控制行列
    #         new_adj1 = create_adj_avg(adj)
    #         for i in range(5):
    #             embedding1 = IK_fm_dot(feat,psi=64,t=100)
    #             embedding1 = WL_noconcate_fast(embedding1,new_adj1)
    #             emb1 = preprocessing.normalize(embedding1, norm='l2', axis=1)
    #
    #             acc, nmi1, f1, para, predict_labels = cmd.sc_linear(emb1, 1, num_of_class, true_lab)
    #             print(nmi1)
    #
    #
    #
    #
    #
    #
    #
