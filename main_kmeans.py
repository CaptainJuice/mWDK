import time
from collections import defaultdict
import scipy.io as sio
import numpy as np
import math
import sklearn.cluster as sc
from utils import load_data
import clustering_metric as cm
from utils import  WL_noconcate,IGK_WL_noconcate,WL_noconcate_gcn,IK_fm_dot,create_adj_avg_gcn,WL
import warnings
import scipy.sparse as sp
warnings.filterwarnings('ignore')




if __name__ == '__main__':

    path1 = 'E:/Graph Clustering/dataset/real_world data/'
    save_path = 'E:/Graph Clustering/dataset/embedding'
    # datasets1 = [ 'dblp', 'amap', 'eat', 'pubmed']
    datasets1 = ['cora','citeseer','wiki']
    embedding_type='ikwl'
    psi_li =[4,8,16,32,64,128]
    for dataset in datasets1:
        adj_mat, node_features,true_labels = load_data(path1,dataset)
        num_of_class = len(list(set(true_labels)))
        rep = 10
        print("========================= {}_{} ============================".format(dataset,embedding_type+'_km'))

        list_nmi, list_f1, list_acc = {},{},{}
        best_nmi,best_f1,best_acc,best_h=-1,-1,-1,-1

        if embedding_type=='ikwl':

            for psi in psi_li:

                features = IK_fm_dot(node_features, psi , 100)
                for h in range(1,30):
                    local_nmi, local_f1, local_acc = [], [], []
                    embedding = WL_noconcate(features, adj_mat, h)
                    savepath = '{}/{}/{}_{}_psi_{}_h_{}_'.format(save_path, dataset, dataset, 'ikwl', psi, h)
                    data = {'data': embedding, 'class': true_labels}
                    sio.savemat('{}.mat'.format(savepath), data)
                    for r in range(rep):
                        # u, s, v = sp.linalg.svds(embedding, k=num_of_class, which='LM')
                        predict_labels = sc.KMeans(n_clusters=num_of_class).fit_predict(embedding)
                        # predict_labels = sc.SpectralClustering(n_clusters=num_of_class,affinity='linear').fit_predict(embedding)
                        translate, new_predict_labels = cm.translate(true_labels, predict_labels)
                        acc, nmi, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, new_predict_labels)
                        local_acc.append(acc)
                        local_nmi.append(nmi)
                        local_f1.append(f1_macro)

                    list_nmi[h] = np.mean(local_nmi)
                    list_f1[h] = np.mean(local_f1)
                    list_acc[h] = np.mean(local_acc)
                    print('@psi={} h={}: ACC:{:.6f}(std:{:.6f})  NMI:{:.6f}(std:{:.6f})  f1_macro:{:.6f}(std:{:.6f})'.format(psi, h, np.mean(local_acc), np.std(local_acc), np.mean(local_nmi), np.std(local_nmi), np.mean(local_f1),np.std(local_f1)))

                print("@BEST h={} ACC:{:.6f} NMI:{:.6f}  f1_macro:{:.6f}".format(max(list_nmi, key = list_nmi.get),max(list_acc.values()),max(list_nmi.values()),max(list_f1.values())))



        # if embedding_type=='ikwl':
        #     save_path = 'E:/Graph Clustering/dataset/embedding'
        #     list_acc,list_nmi,list_f1=[],[],[]
        #     psi_li = [8,16,32,64,128]
        #
        #     t = 150
        #     begin = time.perf_counter()
        #     for psi in psi_li:
        #         dict_nmi, dict_f1, dict_acc = defaultdict(list), defaultdict(list), defaultdict(list)
        #
        #         for r in range(rep):
        #             node_features = IK_fm_dot(node_features, t, psi)
        #             for h in range(1,45):
        #                 embedding = WL(node_features,adj_mat,h)
        #                 savepath = '{}/{}/{}_{}_psi_{}_h_{}_r_{}'.format(save_path, dataset, dataset, 'ikgc', psi, h,r)
        #                 data = {'data': embedding, 'class': true_labels}
        #                 sio.savemat('{}.mat'.format(savepath), data)
        #                 predict_labels = sc.KMeans(n_clusters=num_of_class).fit_predict(embedding)
        #                 translate, new_predict_labels = cm.translate(true_labels, predict_labels)
        #                 acc, nmi, f1, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, new_predict_labels)
        #                 dict_nmi[h].append(nmi)
        #                 dict_acc[h].append(acc)
        #                 dict_f1[h].append(f1)
        #         for h in range(1,45):
        #             nmi=np.mean(dict_nmi[h])
        #             acc=np.mean(dict_acc[h])
        #             f1=np.mean(dict_f1[h])
        #             nmi_std=np.std(dict_nmi[h])
        #             acc_std=np.std(dict_acc[h])
        #             f1_std=np.std(dict_f1[h])
        #             list_nmi.append(nmi)
        #             list_f1.append(f1)
        #             list_acc.append(acc)
        #             print('@psi={} h={}:  ACC:{:.6f}({:.6f})  NMI:{:.6f}({:.6f})  f1_macro:{:.6f}({:.6f})'.format(psi,h,acc,acc_std,nmi,nmi_std,f1,f1_std))
        #     end = time.perf_counter()
        #     print('Best:  ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f} '.format(np.max(list_acc),np.max(list_nmi),np.max(list_f1)))
        #     print('Total Time Cost: {}s'.format(end-begin))
        #
        #
