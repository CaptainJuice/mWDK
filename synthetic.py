import random
from sklearn import preprocessing
import networkx as nx
import warnings
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
import numpy as np
import os
import heapq
from utils import adj_plot
warnings.filterwarnings('ignore')


def Gaussian_Distribution(mu, sigma, dim, num_nodes):   #dim,nodes_num, mu, sigma
    mean = np.zeros(dim)+np.random.rand(dim)+mu
    cov = np.eye(dim) * sigma

    # nums = [i for i in range(20)]
    # weights = [0.1*random.randint(2,6) for i in range(20)]
    # for i in range(cov.shape[0]):
    #     for j in range(i,cov.shape[0]):
    #         cov[i][j] = random.choices(nums,weights=weights,k=1)[0]
    #         cov[j][i] = cov[i][j]

    data = np.random.multivariate_normal(mean, cov, num_nodes)

    return data
data = '1'
dirs = "E:/Graph Clustering/dataset/artificial data/raw_data/Graph_{}".format(data)
if not os.path.exists(dirs):
    os.makedirs(dirs)
#################################################################################################################
dim = 2
nodes_num_li = [1000,1000]
average_degree = [3,3]
#################################################################################################################

num_of_class = len(nodes_num_li)
num_of_nodes = np.sum(nodes_num_li)
ind = [i for i in range(num_of_nodes)]
true_labels = [0 if i<nodes_num_li[0] else 1 for i in range(num_of_nodes)]
node_features=[]
mu_li = [0,0.7]
sigma_li = [0.4,0.4]
for i in range(len(nodes_num_li)):
    mu = mu_li[i]
    sigma = sigma_li[i]
    nums_nodes = nodes_num_li[i]
    temp = Gaussian_Distribution(dim=dim, mu=mu,sigma = sigma,num_nodes=nums_nodes)
    node_features.extend(temp)
node_features = np.array(node_features)
true_labels = np.array(true_labels)
# node_features -= np.min(node_features,axis=0)
# # #
tsne = TSNE(n_components=2,perplexity=35,learning_rate='auto',init='pca')
node_features_tsne = tsne.fit_transform(node_features)
plt.scatter(node_features[:,0], node_features[:,1],s=4, c=true_labels, cmap="rainbow")
# plt.show()


plt.title('Visualization of Graph-{}(t-SNE)'.format(data))
plt.savefig(dirs+'/tsne.png')
plt.show()
############################### adj_mat #########################################

adj_mat = np.zeros((num_of_nodes,num_of_nodes),dtype=float)
dis = pdist(node_features, 'euclidean')
dis = squareform(dis)

for i in range(len(nodes_num_li)):
    cluster = np.where(true_labels==i)[0]
    temp_node_features = node_features[cluster]
    dis = pdist(temp_node_features, 'euclidean')
    dis = squareform(dis)
    for j in cluster:
        degree_turb = np.array([i for i in range(-average_degree[i] + 2, average_degree[i] * 3)])
        prob = np.array([max(1 - 0.15 * i * np.sqrt(i), 0.01) for i in range(len(degree_turb))])
        prob = prob / np.sum(prob)
        turb = np.random.choice(a=degree_turb, size=1, replace=True, p=prob)[0]

        p = average_degree[i] + turb
        q=j
        if j>=nodes_num_li[0]:
            q=j-nodes_num_li[0]

        k_ind = heapq.nsmallest(p, range(len(dis[q])), dis[q].take)

        k_ind =[cluster[i] for i in k_ind]

        for k in k_ind:

            if j != k: adj_mat[j][k], adj_mat[k][j] = 1, 1
deg = np.sum(adj_mat,axis=0)
mean_degree = np.mean(deg)
adj_plot(adj_mat,true_labels,100)
cnt, cmt = 0, 0
for i in range(node_features.shape[0]):
    node_degree = int(deg[i])
    p_inner = 0.5
    if node_degree==1:
        p_inter = 0
    elif node_degree<mean_degree:
        p_inter = 0.1
    elif node_degree<mean_degree+2:
        p_inter = 0.2
    else:
        p_inter = 0.4

    inner = np.random.choice(a=[0,1], size=1, replace=True, p=[1-p_inner, p_inner])[0]
    inner = 1
    if inner==1 and i < nodes_num_li[0]:
        adj_mat[i] = 0
        ind_li = ind[:nodes_num_li[0]]
        j = np.random.choice(a=ind_li, size=node_degree, replace=True, p=[1/len(ind_li)]*len(ind_li))
        adj_mat[i][j], adj_mat[j][i] = 1,1

    #
    # for j in range(node_features.shape[0]):
    #     if (i < nodes_num_li[0] and j > nodes_num_li[0]) or (i > nodes_num_li[0] and j < nodes_num_li[0]):
    #         if s == 1:
    #             t = random.randint(1, 120)
    #             if t == 1: adj_mat[i][j], adj_mat[j][i] = 1, 1
    #
    #         if p==1:
    #             t = random.randint(1, 70)
    #             if t == 1: adj_mat[i][j], adj_mat[j][i] = 1, 1
    #         pass
    #
    #     else:
    #         if inner==1:
    #             node_degree
    #
    #
    #
    #
    #
    #         if adj_mat[i][j] == 1 and inner > 500 :
    #             cnt +=1
    #             adj_mat[i][j], adj_mat[j][i] = 0, 0
    #         elif adj_mat[i][j] == 0 and inner <3:
    #             cmt+=1
    #             adj_mat[i][j], adj_mat[j][i] = 1, 1
print("del:{} add:{}".format(cnt,cmt))
#
# for i in range(len(adj_mat)):
#
#     G.add_node(i, name = i)
#
# Matrix = adj_mat
# for i in range(len(Matrix)):
#     for j in range(len(Matrix)):
#         if Matrix[i][j] == 1:
#             G.add_edge(i, j)
#
plt.rcParams['axes.facecolor'] = 'snow'
adj_plot(adj_mat,true_labels,100)
G = nx.from_numpy_matrix(adj_mat)
# nx.draw(G,node_color=true_labels,node_size=1.5,alpha=0.15)
# nx.draw_networkx_nodes(G, node_features, node_size=1, node_color=true_labels)  # 画节点
# nx.draw_networkx_edges(G, node_features, alpha=0.3, width=1)  # 画边
# nx.draw_networkx_labels(G, X, node_labels=label,font_size=20)
# nx.draw_networkx_nodes(G, npos, node_size=5, node_color=label)  # 绘制节点
# nx.draw_networkx_edges(G, npos, adj_mat)  # 绘制边
# nx.draw_networkx_labels(G, npos, nlabels)  # 标签
plt.show()
########################### check #################################
deg = np.sum(adj_mat,axis=0)
if np.any(deg==0): print('Exist degree is 0!')


np.save(dirs+'/Graph_{}_label'.format(data),true_labels)
np.save(dirs+'/Graph_{}_adj'.format(data),adj_mat)
np.save(dirs+'/Graph_{}_feat'.format(data),node_features)
print("@END(Graph-V{}) nodes:{}  edges:{}  degree1:{}  degree2:{} ".format(data, G.number_of_nodes(), G.size(), np.sum(adj_mat[:nodes_num_li[0],:nodes_num_li[0]])/2,np.sum(adj_mat[nodes_num_li[0]:,nodes_num_li[0]:])/2,))
