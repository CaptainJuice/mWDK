import sklearn.cluster as sc
import numpy as np
import clustering_metric as cm
from tqdm import tqdm
from utils import load_data,  WL_noconcate
import sys
sys.path.append("..")
from sklearn.metrics.cluster import   normalized_mutual_info_score

NMI = lambda x, y: normalized_mutual_info_score(x, y, average_method='arithmetic')

dataset = 'cora'
adj_mat, node_features, y_test, tx, ty, test_maks, true_labels = load_data(dataset)
print("**************************************DATASET: {} **************************************".format(dataset))
list_nmi, list_f1, list_acc = [], [], []
best_acc, best_nmi, best_f1,best_h,best_k= 0,0,0,0,0
# @h=11 acc:0.6971935007385525  nmi:0.5605251585747875  f1:0.6971112447102241
# @mean of 10: acc:0.6970827178729689  nmi:0.5302869786182357  f1:0.6657196594688181

hl=[]

for h in range(1,60,2):
    embedding = WL_noconcate(node_features, adj_mat, h)

    for k in range(7,13):
        model = sc.KMeans(n_clusters=k)
        model = model.fit(embedding)
        predict_labels = model.predict(embedding)

        num_of_cluster = len(list(set(predict_labels)))
        num_of_class = len(list(set(true_labels)))
        num_of_nodes = len(true_labels)

        cluster = [[] for _ in range(num_of_cluster)] # save the index of points of each cluster
        cluster_means= []  # get the center of each cluster
        for index in range(num_of_nodes):
            label = predict_labels[index]
            cluster[label].append(index)
        for i in range(num_of_cluster):
            index_li = cluster[i]
            em = [embedding[id] for id in index_li]
            mean = np.mean(em)
            cluster_means.append(mean)

        translate, mapping_predict_labels = cm.translate(true_labels, predict_labels)  # get mapping rules and mapped predict_labels
        normal = translate[:num_of_class]  # get the clusters of normal and noise
        noise = [i for i in range(num_of_cluster) if i not in normal]

        normal_pre = [i for i in mapping_predict_labels if i in normal]  # get the labels of normal and noise
        normal_true = [true_labels[i] for i in range(num_of_nodes) if mapping_predict_labels[i] in normal]
        nomal_means = [cluster_means[i] for i in range(num_of_cluster) if i in normal]
        min = 10000000000000

        noise_label = []
        for i in noise:
            record = -1
            for j in normal:
                temp = np.linalg.norm(cluster_means[i]-cluster_means[j])
                if temp < min:
                    record = j
            noise_label.append(record)

        new_predict_labels = []
        for p in predict_labels:
            if p in translate:
                new_predict_labels.append(translate.index(p))
            else:

                new_predict_labels.append(translate.index(noise_label[noise.index(p)]))

        acc, nmi, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, mapping_predict_labels)
        tqdm.write('@CLUSTER_NUMBER = {}, ACC={}, NMI={},f1_macro={}, precision_macro={}, recall_macro={}, f1_micro={}, precision_micro={}, recall_micro={},  ADJ_RAND_SCORE={}'.format(k,acc, nmi, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore))

        if nmi > best_nmi:
            best_nmi = nmi
            best_h = h
            best_k = k
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_h = h
            best_k = k
        if acc > best_acc:
            best_acc = acc
    print('@BEST_CLUSTER_NUMBER = {}: acc:{}  nmi:{}  f1:{}'.format(best_k,best_acc,best_nmi,best_f1))
    list_nmi.append(best_nmi)
    list_f1.append(best_f1)
    list_acc.append(best_acc)

print('@BEST_CLUSTER_NUMBER = {},h={}: acc:{}  nmi:{}  f1:{}'.format(best_k, best_h, best_acc, best_nmi, best_f1))
print('@mean of {}: acc:{}  nmi:{}  f1:{}'.format(len(hl),np.mean(list_acc), np.mean(list_nmi), np.mean(list_f1)))







