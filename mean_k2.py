import sklearn.cluster as sc
from clustering_metric import clustering_metrics
from tqdm import tqdm
from utils import load_data
from utils import  WL
import sys
sys.path.append("..")
from sklearn.metrics.cluster import  normalized_mutual_info_score
from sklearn import metrics
NMI = lambda x, y: normalized_mutual_info_score(x, y, average_method='arithmetic')

dataset = 'cora'
adj_mat, node_features, y_test, tx, ty, test_maks, true_labels = load_data(dataset)
print("========================= {} ============================".format(dataset))
list_nmi, list_f1, list_acc = [], [], []
best_acc, best_nmi, best_f1,best_h= 0, 0, 0,0

hl=[7,11,13]
for h in hl:
    embedding = WL(node_features, adj_mat, h)

    for k in range(8,13):
        model = sc.KMeans(n_clusters=k)
        model = model.fit(embedding)
        predict_labels = model.predict(embedding)

        num_of_pre_class = len(list(set(predict_labels)))
        num_of_class = len(list(set(true_labels)))
        num_of_nodes = len(true_labels)

        cluster = [[] for _ in range(num_of_pre_class)]
        cluster2 =[[] for _ in range(num_of_pre_class)]
        cluster_means= []
        for index in range(num_of_nodes):
            label = predict_labels[index]
            cluster[label].append(index)
            cluster2[label].append(true_labels[index])

        for i in range(len(cluster)):
            mean =max(set(cluster2[i]),key=cluster2[i].count)
            cluster_means.append(mean)
        print(cluster_means)




        cm = clustering_metrics(true_labels, predict_labels)
        translate,new_predict_labels,acc, nmi, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(
            tqdm)

        normal = translate[:num_of_class]
        noise = [i for i in range(num_of_pre_class) if i not in  normal ]
        print(normal,noise)

        normal_pre = [i for i in new_predict_labels if i in normal]
        normal_true = [true_labels[i] for i in range(num_of_nodes) if new_predict_labels[i] in normal]

        nomal_means = [cluster_means[i] for i in range(num_of_pre_class) if i in normal]
        min = 10000000000000
        print(translate)
        print(predict_labels.tolist())
        print(new_predict_labels)

        noise_label = []
        for i in noise:
            print(cluster2[i])
            l = max(set(cluster2[i]), key=cluster2[i].count)
            if l in cluster_means:
                ind = cluster_means.index(l)
            noise_label.append(ind)


        new_predict_labels = []
        for p in predict_labels:
            if p in translate:
                new_predict_labels.append(translate.index(p))
            else:

                new_predict_labels.append(translate.index(noise_label[noise.index(p)]))



        print('@', k,len(normal_pre))
        print('origin acc:{} nmi:{} f1:{}'.format(acc,nmi,f1_macro))
        acc = metrics.accuracy_score(true_labels,new_predict_labels)
        nmi = NMI(true_labels,new_predict_labels)
        f1_macro = metrics.f1_score(true_labels,new_predict_labels,average='macro')
        print('total acc:{} nmi:{} f1:{}'.format(acc,nmi,f1_macro))
        print('nomal acc:',metrics.accuracy_score(normal_true,normal_pre))
        print('nomal nmi:',NMI(normal_true,normal_pre))


        # print('true_labels:',true_labels.tolist())
        # print('predict_labels:',predict_labels.tolist())
        # print('new_labels:',new_predict_labels)
        # print(translate)
        # print(new_predict_labels)
        # print(predict_labels.tolist())






