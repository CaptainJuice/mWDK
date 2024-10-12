import argparse
import numpy as np
from sklearn.cluster import KMeans
from utils import load_data
import clustering_metric as cm
from utils import   WL_noconcate
import warnings
from sklearn.metrics.pairwise import euclidean_distances
import scipy.sparse as sp
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pubmed', help='type of dataset.')# cora citeseer pubmed
parser.add_argument('--embedding-type', type=str, default='wl', help="type of embedding")
args = parser.parse_args()

def normalize_adj(adj, type='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        # d_inv_sqrt = np.power(rowsum, -0.5)
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # return adj*d_inv_sqrt*d_inv_sqrt.flatten()
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized


def preprocess_adj(adj, type='sym', loop=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj, type=type)
    return adj_normalized


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

def dist(prelabel, feature):
    k = len(np.unique(prelabel))
    intra_dist = 0

    for i in range(k):
        Data_i = feature[np.where(prelabel == i)]

        Dis = euclidean_distances(Data_i, Data_i)
        n_i = Data_i.shape[0]
        if n_i == 0 or n_i == 1:
            intra_dist = intra_dist
        else:
            intra_dist = intra_dist + 1 / k * 1 / (n_i * (n_i - 1)) * sum(sum(Dis))


    return intra_dist
if __name__ == '__main__':
    dataset = args.dataset
    embedding_type = args.embedding_type

    adj_mat, node_features, y_test, tx, ty, test_maks, true_labels = load_data(dataset)

    num_of_class = len(list(set(true_labels)))
    print("========================= {}_{} ============================".format(dataset,embedding_type))
    list_nmi, list_f1, list_acc = [], [], []
    best_acc, best_nmi, best_f1,best_h= 0, 0, 0,0

    if embedding_type=='wl':
        rep = 10
        adj_normalized = preprocess_adj(np.array(adj_mat[0]))
        adj_normalized = (sp.eye(adj_normalized.shape[0]) + adj_normalized) / 2
        for h in range(1,20,2):
            embedding = WL_noconcate(node_features, adj_mat, h)

            feature = adj_normalized.dot(embedding)
            u, s, v = sp.linalg.svds(feature, k=num_of_class, which='LM')
            kmeans = KMeans(n_clusters=num_of_class).fit(u)
            predict_labels = kmeans.predict(u)
            translate, new_predict_labels = cm.translate(true_labels, predict_labels)
            acc, nmi, f1, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, new_predict_labels)
            current_acc =  acc
            current_nmi=nmi
            current_f1 = f1
            print("@h={} Acc: ".format(h),acc," NMI: ", nmi, " F1: ",f1)
            if current_nmi > best_nmi:
                best_nmi = current_nmi
                best_h = h
            if current_f1 > best_f1:
                best_f1 = current_f1
            if current_acc > best_acc:
                best_acc = current_acc
        print("Best_NMI:", best_nmi, " @h:", best_h )
        for i in range(rep):
            embedding = WL_noconcate(node_features, adj_mat, best_h)
            feature = adj_normalized.dot(embedding)
            u, s, v = sp.linalg.svds(feature, k=num_of_class, which='LM')
            kmeans = KMeans(n_clusters=num_of_class).fit(u)
            predict_labels = kmeans.predict(u)
            translate, new_predict_labels = cm.translate(true_labels, predict_labels)
            acc, nmi, f1, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(
                true_labels, new_predict_labels)
            list_nmi.append(nmi)
            list_f1.append(f1)
            list_acc.append(acc)

        print("NMI:  Mean:{}  Max:{}  Min:{}".format(np.mean(list_nmi), np.max(list_nmi), np.min(list_nmi)))
        print("ACC:  Mean:{}  Max:{}  Min:{}".format(np.mean(list_acc), np.max(list_acc), np.min(list_acc)))
        print("F1:  Mean:{}  Max:{}  Min:{}".format(np.mean(list_f1), np.max(list_f1), np.min(list_f1)))

        print("Saving successfully")