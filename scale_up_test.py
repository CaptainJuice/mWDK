import time
import numpy as np
from utils import IK_fm_dot
from utils import load_data, GSNN, GSNN_try, pplot2, create_adj_avg_gcn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import clustering_methods as cmd
import sklearn.cluster as sc
import scipy
import scipy.sparse as sp
import warnings
from sklearn import preprocessing
import random
import torch
import sys
sys.path.append("E:\Graph Clustering\ik_mkVI_aNNE\ik_mkVI\gpu")
# from gpu_utils import gpu_init,gpu_deinit
# from ik import fit, transform
warnings.filterwarnings('ignore')
from ogb.nodeproppred import NodePropPredDataset

import copy
def IK_fm_dot_sp(X,psi,t,):

    onepoint_matrix = np.zeros((X.shape[0], (int)(t * psi)), dtype=int)
    onepoint_matrix = sp.csr_matrix(onepoint_matrix)
    x_index=np.arange(X.shape[0])
    for time in range(t):
        sample_num = psi  #
        sample_list = [p for p in range(X.shape[0])]  # [0, 1, 2, 3]
        sample_list = random.sample(sample_list, sample_num)  # [1, 2]
        sample = X[sample_list, :]  # array([[ 4,  5,  6,  7], [ 8,  9, 10, 11]])
        # sim
        point2sample =np.dot(X,sample.T)
        min_dist_point2sample = np.argmax(point2sample, axis=1)+time*psi
        min_dist_point2sample = np.array(min_dist_point2sample).reshape(1,-1)[0]
        data = [1 for _ in range(min_dist_point2sample.shape[0])]
       # dis
       #  from sklearn.metrics.pairwise import euclidean_distances
       #  point2sample =euclidean_distances(X,sample)
       #  min_dist_point2sample = np.argmin(point2sample, axis=1)+time*psi
        onepoint_matrix = sp.csr_matrix((data, (x_index, min_dist_point2sample)))

    return onepoint_matrix
def create_adj_avg(adj_mat):
    '''
    create adjacency
    '''
    np.fill_diagonal(adj_mat, 0)

    adj = copy.deepcopy(adj_mat)
    deg = np.sum(adj, axis=1)
    deg[deg == 0] = 1
    deg = (1/ deg) * 0.5
    deg_mat = np.diag(deg)
    adj = deg_mat.dot(adj_mat)
    np.fill_diagonal(adj, 0.5)
    return adj
def create_adj_avg_sp(adj_mat):
    '''
    create adjacency
    '''
    adj = copy.deepcopy(adj_mat)

    deg = np.array(sp.csr_matrix.sum(adj, axis=1)).reshape(-1)
    deg = (1 / deg) * 0.5
    deg_mat = sp.diags(deg)
    adj = deg_mat * adj
    adj.setdiag(0.5)
    return adj


def WL_noconcate_fast(embedding, adj_mat):
    new_embedding = adj_mat * embedding

    return new_embedding


dataset = NodePropPredDataset(name='ogbn-products')

g, true_labels = dataset[0]
node_features = g['node_feat']
row = g['edge_index'][0]
col = g['edge_index'][1]
data = np.ones_like(row)

# labels = copy.deepcopy(true_labels)
# path1 = 'E:/Graph Clustering/dataset/real_world data/'
# dataset='pubmed'
# adj_mat, node_features, true_labels = load_data(path1, dataset)
# adj_mat = sp.csr_matrix(adj_mat)

node_features =scipy.sparse.csc_matrix(node_features)
# scipy.sparse.save_npz('adj',adj_mat)
scipy.sparse.save_npz('feature',node_features)
np.save('labels',true_labels)

print('ok')
#
adj_mat = scipy.sparse.load_npz('adj.npz', )
node_features = np.load('feature.npy', allow_pickle=True)
true_labels = np.load('labels.npy', allow_pickle=True).reshape(-1)
node_features = sp.csr_matrix(node_features)
# rep = 1
###################################################################################


num_of_class = np.unique(true_labels).shape[0]
print(num_of_class,node_features.shape[1])
new_adj = create_adj_avg_sp(adj_mat)

for h in range(15):
    embedding = WL_noconcate_fast(embedding, new_adj)

    acc, nmi, f1, para, predict_labels = cmd.km(embedding, 1, num_of_class, true_labels)
    print('@{} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))

# adj_mat = scipy.sparse.load_npz('adj.npz',).todense()
# node_features = np.load('feature.npy',allow_pickle=True)
# true_labels = np.load('labels.npy',allow_pickle=True)
# #


node_features = sp.coo_matrix(node_features)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
node_features = sparse_mx_to_torch_sparse_tensor(node_features)
print('ok')
num_of_class = np.unique(true_labels).shape[0]
# acc_li, nmi_li, f1_li = [], [], []
print(adj_mat.shape, node_features.shape,true_labels.shape)
print(np.sum(adj_mat)/2,num_of_class)
length = adj_mat.shape[0]
# scale_li=[int(length*0.2),int(length*0.4),int(length*0.5),int(length*0.8),length]
# scale_li =[1000000]

best_acc, best_nmi, best_f1, best_h = -1, -1, -1, -1

time_start = time.perf_counter()
# long running
# do something other
adj = adj_mat
embedding = node_features
new_adj = create_adj_avg(adj)
# embedding = IK_fm_dot(embedding, 32, 100)
myli = [i for i in range(node_features.shape[0])]
config = {'alg': 'aNNE', 'dist_func': 'linear', 'p_value': 10, 'memGPU': 4, 'seed': 42, 'nInst': embedding.shape[0], 'nAttr': embedding.shape[1], 'nSets': 100, 'nPsi': 8}

for h in range(8):
    # mdl = fit(embedding, config)
    # embedding = transform(embedding, mdl, "feat")  # (sim)ilarity or (feat)ure space
    embedding = IK_fm_dot(embedding, 32, 100)
    embedding = preprocessing.normalize(embedding, norm='l2', axis=0 )
    embedding = WL_noconcate_fast(embedding, new_adj)
    emb = preprocessing.normalize(embedding, norm='l2', axis=1 )
    predict_labels = sc.SpectralClustering(n_clusters=num_of_class, affinity='linear').fit_predict(embedding)

    time_end = time.perf_counter()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

    print(nmi)
