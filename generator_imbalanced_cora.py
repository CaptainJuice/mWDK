import copy

import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import random
from utils import adj_plot


def adjConcat(a, b):
    '''
    将a,b两个矩阵沿对角线方向斜着合并，空余处补零[a,0.0,b]
    得到a和b的维度，先将a和b*a的零矩阵按行（竖着）合并得到c，再将a*b的零矩阵和b按行合并得到d
    将c和d横向合并
    '''
    lena = a.shape[0]
    lenb = b.shape[0]
    left = np.row_stack((a, np.zeros((lenb, lena))))  # 先将a和一个len(b)*len(a)的零矩阵垂直拼接，得到左半边
    right = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个len(a)*len(b)的零矩阵和b垂直拼接，得到右半边
    result = np.hstack((left, right))  # 将左右矩阵水平拼接
    return result


def Gaussian_Distribution(N, M, mu, sigma):   #dim,nodes_num, mu, sigma
    mean = np.zeros(N) + mu
    cov = np.eye(N) * sigma
    # nums = [i for i in range(20)]
    # weights = [0.1*random.randint(2,6) for i in range(20)]
    # for i in range(cov.shape[0]):
    #     for j in range(i,cov.shape[0]):
    #         cov[i][j] = random.choices(nums,weights=weights,k=1)[0]
    #         cov[j][i] = cov[i][j]

    data = np.random.multivariate_normal(mean, cov, M)
    return data
data = 'imbalanced_edges_from_cora'
dirs = "E:/Graph Clustering/dataset/artificial data/raw_data/Graph_{}".format(data)
if not os.path.exists(dirs):
    os.makedirs(dirs)
print("Genarating ···".format(data))

#################################################################################################################

# X = []
# y= []
# for id in range(len(nodes_num_li)):
#     mu = mu_li[id]
#     sigma = sigma_li[id]
#     nodes_num = nodes_num_li[id]
#     x =Gaussian_Distribution(dim,nodes_num, mu, sigma)
#     label = [id]*nodes_num
#     X.extend(x)
#     y.extend(label)
# X =np.array(X)
# # embedding= (X-np.min(X))/(np.max(X)-np.min(X))
# embedding=X
# true_labels=np.array(y)
# from sklearn.metrics.pairwise import pairwise_distances
# from sklearn import preprocessing
# sim = pairwise_distances(embedding,metric='euclidean')
# sim= (sim-np.min(sim))/(np.max(sim)-np.min(sim))
# # sim = preprocessing.normalize(sim, norm='l2',axis=0)
#
#
# adj_mat = np.zeros((nodes_num_li[0]*2,nodes_num_li[0]*2))
#
# for i in range(num_of_nodes):
#     for j in range(num_of_nodes):
#         if (i < nodes_num_li[0] and j > nodes_num_li[0]) or (i > nodes_num_li[0] and j < nodes_num_li[0]):
#             a = random.randint(1,int(sim[i][j]*700))
#             if a==1:
#                 adj_mat[i][j],adj_mat[j][i]=1,1
#         elif i < nodes_num_li[0]:
#             a = random.randint(1,10)
#             if a==1:
#                 adj_mat[i][j],adj_mat[j][i]=1,1
#         else:
#             a = random.randint(1,20)
#             if a==1:
#                 adj_mat[i][j],adj_mat[j][i]=1,1
from utils import load_data

path1 = 'E:/Graph Clustering/dataset/real_world data/'
datasets1 = ['cora', 'citeseer', 'pubmed', 'amap', ]
datasets2 = ['blogcatalog', 'flickr', 'wiki', 'dblp', 'acm']
# path1 = 'E:/Graph Clustering/dataset/artificial data/'
# datasets = ['cora']
dataset = 'cora'
####################################################################################

adj_mat, node_features, true_labels = load_data(path1, dataset)
temp_li=[]
for i,j in enumerate(true_labels):
    if j==1 or j ==4:
        temp_li.append(i)


i, j = np.ix_(temp_li,temp_li)

adj_mat=adj_mat[i,j]
node_features=node_features[temp_li]
true_labels=true_labels[temp_li]
true_labels = np.where(true_labels==0,0,true_labels)
true_labels = np.where(true_labels==4,1,true_labels)
true_labels = np.where(true_labels==5,2,true_labels)
one_li = [i for i,j in enumerate(true_labels) if j==1]
# for i in one_li:
#     for j in one_li:
#         a = random.randint(1,300)
#         if a==10 and i!=j:
#             adj_mat[i][j],adj_mat[j][i]=1,1
# for i in range(len(true_labels)):
#     for j in range(len(true_labels)):
#         if (i in one_li and j not in one_li):
#                 a = random.randint(1,50)
#                 if a==1:
#                     a= random.randint(1,100)
#                     if a== 1:
#                         adj_mat[i][j],adj_mat[j][i]=1,1


tsne = TSNE(n_components=2, perplexity=55)
node_features_tsne = tsne.fit_transform(node_features)
adj_plot(adj_mat,true_labels,50)
plt.scatter(node_features_tsne[:, 0], node_features_tsne[:, 1],s=5, c=true_labels, cmap="rainbow")
plt.title('raw:Graph-{}'.format(data))
plt.savefig(dirs+'/raw.png')
plt.show()


np.save(dirs+'/graph_{}_label'.format(data),true_labels)
np.save(dirs+'/graph_{}_adj'.format(data),adj_mat)
np.save(dirs+'/graph_{}_feat'.format(data),node_features)

G = nx.from_numpy_matrix(adj_mat)
#


pos = node_features_tsne

nx.draw_networkx_nodes(G, pos, node_size=3, node_color=true_labels)  # 画节点
nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)  # 画边
plt.show()
adj_plot(adj_mat,true_labels,50)

