import numpy as np
import clustering_methods as cmd
from utils import WL, load_data, WL_noconcate, IGK_WL_noconcate,IK_fm_dot,WL_noconcate_gcn,pplot2,knn_WL_noconcate
import warnings
import scipy.io as sio
warnings.filterwarnings('ignore')
from sklearn import preprocessing
import networkx as nx






if __name__ == '__main__':

    ###################################### parm ########################################
    emb_type = {"wl_noconcate": 1,
              "ikwl_noconcate": 0}
    path1 = 'E:/Graph Clustering/dataset/real_world data/'
    datasets1 = ['cora','citeseer','wiki','acm','dblp','amap','eat']
    path2 = 'E:/Graph Clustering/dataset/artificial data/'
    datasets = ["cora"]
    rep = 1
    ####################################################################################

    if emb_type['wl_noconcate'] == 1:
        for dataset in datasets:
            adj_mat, node_features, true_labels = load_data(path1, dataset)
            num_of_class = np.unique(true_labels).shape[0]
            acc_li,nmi_li,f1_li = [],[],[]
            for r in range(rep):
                best_acc, best_nmi, best_f1, best_h= -1, -1, -1, -1
                features_IK = IK_fm_dot(node_features, 300, 64)
                sim = features_IK.dot(features_IK.T)

                # np.fill_diagonal(sim,0)
                # sim = sim / np.max(sim)
                # sim = preprocessing.normalize(sim, norm='l2')
                # sim = sim*adj_mat
                adj_h = adj_mat
                for t in range(3):
                    adj_h = adj_h.dot(adj_mat)+adj_h
                    # np.fill_diagonal(adj_h,0)
                for h in range(1,19):
                    embedding = WL_noconcate(node_features,adj_h,h)
                    # embedding = WL_noconcate(features_IK, adj_h, h)
                    embedding = preprocessing.normalize(embedding, norm='l2')

                    acc,nmi,f1,para,predict_labels = cmd.sc_linear(embedding,1,num_of_class,true_labels)
                    print('@h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(h,para,acc,nmi,f1))
                    if best_nmi < nmi:
                        best_nmi = nmi
                        best_h = h
                    if best_f1 < f1:
                        best_f1 = f1
                    if best_acc < acc:
                        best_acc = acc
                if best_nmi==1:
                    print("Perfect!!!")
                    break
                print('@BEST(r= {} IKWL-{}) (h={}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(r+1,para,best_h,best_acc,best_nmi,best_f1))
                acc_li.append(best_acc)
                nmi_li.append(best_nmi)
                f1_li.append(best_f1)
            print('@BEST(rep= {} IKWL-{}) (h={}): ACC:{:.6f}({:.6f})  NMI:{:.6f}({:.6f})  f1_macro:{:.6f}({:.6f})'.format(rep, para, best_h, np.mean(acc_li),np.std(acc_li),np.mean(nmi_li),np.std(nmi_li), np.mean(f1_li),np.std(f1_li)))

