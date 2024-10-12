import argparse
import numpy as np
import sklearn.cluster as sc
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn import preprocessing
from utils import load_data,gsnn_adj,WL_noconcate
import clustering_metric as cm
from utils import WL
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pubmed', help='type of dataset.')# cora citeseer pubmed
parser.add_argument('--embedding-type', type=str, default='wl', help="type of embedding")
args = parser.parse_args()

#0.499
if __name__ == '__main__':
    rep = 5
    dataset = args.dataset
    embedding_type = args.embedding_type
    adj_mat, node_features,true_labels = load_data(dataset)
    adj_mat =gsnn_adj(adj_mat,1)

    num_of_class = len(list(set(true_labels)))
    print("========================= {}_{} ============================".format(dataset,embedding_type))
    list_nmi, list_f1, list_acc = [], [], []
    best_acc, best_nmi, best_f1,best_h,best_gamma= -1,-1,-1,-1,-1
    gamma_li = [0.0001,0.001,0.01,0.1,0.5,1,]
    if embedding_type=='wl':
        for h in range(1,25,2):
            embedding = WL_noconcate(node_features, adj_mat, h)
            local_acc, local_nmi, local_f1, local_gamma = -1,-1,-1,-1
            for gamma in gamma_li:
                predict_labels = sc.SpectralClustering(n_clusters=num_of_class, gamma=gamma,affinity='linear').fit_predict(embedding)
                translate, new_predict_labels = cm.translate(true_labels, predict_labels)
                acc, nmi, f1, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, new_predict_labels)

                print('@h={} (gamma={:.3f}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(h, gamma, acc, nmi, f1))

                if nmi > local_nmi:
                    local_nmi = nmi
                    local_gamma = gamma
                if f1 > local_f1:
                    local_f1 = f1
                if acc > local_acc:
                    local_acc = acc

            if local_nmi > best_nmi:
                best_nmi = local_nmi
                best_gamma = local_gamma
                best_h = h
            if local_f1 > best_f1:
                best_f1 = local_f1
            if local_acc > best_acc:
                best_acc = local_acc
        for i in range(rep):
            embedding = WL_noconcate(node_features, adj_mat, best_h)
            predict_labels = sc.SpectralClustering(n_clusters=num_of_class, gamma=best_gamma).fit_predict(embedding)
            translate, new_predict_labels = cm.translate(true_labels, predict_labels)
            acc, nmi, f1, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, new_predict_labels)
            list_nmi.append(nmi)
            list_f1.append(f1)
            list_acc.append(acc)
        print('@BEST(WL) (h={} gamma={:.3f}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(best_h, best_gamma,np.max(list_acc),np.max(list_nmi),np.max(list_f1)))
        print('@MEAN OF {}(WL): ACC:{:.6f}(std:{:.6f})  NMI:{:.6f}(std:{:.6f})  f1_macro:{:.6f}(std:{:.6f})'.format(rep,np.mean(list_acc),np.std(list_acc),np.mean(list_nmi),np.std(list_nmi),np.mean(list_f1),np.std(list_f1)))



