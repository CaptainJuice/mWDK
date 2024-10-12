import sklearn.cluster as sc
from clustering_metric import clustering_metrics
import numpy as np
from tqdm import tqdm
from utils import load_data
from utils import  WL
from run import main
import pandas as pd
import random
import sys
sys.path.append("..")

dataset = 'cora'

adj_mat, node_features, y_test, tx, ty, test_maks, true_labels = load_data(dataset)

print("========================= {} ============================".format(dataset))
list_nmi, list_f1, list_acc = [], [], []
best_acc, best_nmi, best_f1,best_h= 0, 0, 0,'ERROR'

ac_l = [0.01]
h_l = [13]
for h in h_l:
    embedding = WL(node_features, adj_mat, h)

    model = sc.KMeans(n_clusters=len(list(set(true_labels))))
    model = model.fit(embedding)
    predict_labels = model.predict(embedding)

    # result_list, centerIndex_list = DensityPeak(X=embedding, n_cluster=len(list(set(true_labels)))).autoSearch()
    # for predict_labels in result_list:
    cm = clustering_metrics(true_labels, predict_labels)
    acc,nmi, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(tqdm)
    current_acc = acc
    current_nmi=nmi
    current_f1 = f1_macro

    if current_nmi > best_nmi:
        best_nmi = current_nmi
        best_h = h
        best_labels = predict_labels
    if current_f1 > best_f1:
        best_f1 = current_f1
    if current_acc > best_acc:
        best_acc = current_acc

    print("@h={}  best_nmi:{}   best_nmi:{}   best_f1:{}".format(best_h,best_acc,best_nmi,best_f1))


for train in range(6):
    predict_labels =best_labels
    adj_mat=np.array(adj_mat[0])
    features=[[] for i in range(len(list(set(true_labels))))]
    edges =[[] for i in range(len(list(set(true_labels))))]
    cluster = [[] for i in range(len(list(set(true_labels))))]
    for id in range(len(predict_labels)):
        label=predict_labels[id]
        edges[label].append(id)

        features[label].append(node_features[0][id])
    adj=[]
    for id in range(len(features)):
        i, j = np.ix_(edges[id],edges[id])

        adj.append(adj_mat[i,j])

    del_li=[]

    for i in range(len(list(set(true_labels)))):


        score = main(features[i],adj[i],1,2).reshape(-1).tolist()[0]
        ac =random.sample(ac_l,1)

        n = int(len(features[i])*ac[0])
        # 选取list数组元素中最大（最小）的n个值的索引
        res = pd.Series(score).sort_values().index[:n].tolist()
        res =[edges[i][j] for j in res ]
        del_li += res
    ll = len(list(set(del_li)))
    rr = len(del_li)
    features2 = node_features[0]
    features2 = np.delete(features2,del_li,axis=0)
    adj = np.delete(adj_mat, del_li, axis=1)
    adj = np.delete(adj, del_li, axis=0)
    node_features2,adj_mat2=[],[]
    node_features2.append(features2)
    adj_mat2.append(adj)
    true_labels2 =np.delete(true_labels, del_li, axis=0)

    best_acc, best_nmi, best_f1,= 0, 0, 0,

    embedding = WL(node_features2, adj_mat2, best_h)
    print(node_features2[0].shape,adj_mat2[0].shape,rr,ll)
    model = sc.KMeans(n_clusters=len(list(set(true_labels))))
    model = model.fit(embedding)
    predict_labels = model.predict(embedding)
    cm = clustering_metrics(true_labels2, predict_labels)
    acc,nmi, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(tqdm)
    current_acc = acc
    current_nmi=nmi
    current_f1 = f1_macro

    if current_nmi > best_nmi:
        best_nmi = current_nmi
        best_h = h
        best_labels = predict_labels
    if current_f1 > best_f1:
        best_f1 = current_f1
    if current_acc > best_acc:
        best_acc = current_acc
    print("@train={} ac={} best_nmi:{}   best_nmi:{}   best_f1:{}".format(train,ac[0],best_acc,best_nmi,best_f1))
    adj_mat =adj_mat2
    node_features = node_features2
    true_labels = true_labels2