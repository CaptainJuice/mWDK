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
data = 'EEE'
dirs = "E:/Graph Clustering/dataset/artificial data/raw_data/{}".format(data)
if not os.path.exists(dirs):
    os.makedirs(dirs)
#################################################################################
# node_feature: gaussian distribution
# adj_mat: random graph
#################################################################################################################
dim = 100
nodes_num_li = [1000,1000]
average_degree = 2
#################################################################################################################

num_of_class = len(nodes_num_li)
num_of_nodes = int(np.sum(nodes_num_li))
true_labels = [0 if i<nodes_num_li[0] else 1 for i in range(num_of_nodes)]
node_features=[]
inner_prob = 0.0058
inter_prob = 0.001
mu_li = [0,1]
sigma_li = [10,10]
for i in range(len(nodes_num_li)):
    mu = mu_li[i]
    sigma = sigma_li[i]
    nums_nodes = nodes_num_li[i]
    temp = Gaussian_Distribution(dim=dim, mu=mu,sigma = sigma,num_nodes=nums_nodes)
    node_features.extend(temp)
node_features = np.array(node_features)
true_labels = np.array(true_labels)
node_features -= np.min(node_features,axis=0)
# # #
tsne = TSNE(n_components=2,perplexity=55,learning_rate=0.00001,init='pca',n_iter=4000)
node_features_tsne = tsne.fit_transform(node_features)
# plt.scatter(node_features[:,0], node_features[:,1],s=4, c=true_labels, cmap="rainbow")
# plt.show()

plt.scatter(node_features[:,0], node_features[:,1],s=5, c=true_labels, cmap="rainbow")
plt.title('Visualization of Graph-{}(t-SNE)'.format(data))
plt.savefig(dirs+'/tsne.png')
plt.show()

############################### adj_mat #########################################


G = nx.random_partition_graph(nodes_num_li,inner_prob,inter_prob)
adj_mat= np.array(nx.adjacency_matrix(G).todense())
adj_plot(adj_mat,true_labels,100)

plt.show()
########################### check #################################
for i in range(adj_mat.shape[0]):
    if np.sum(adj_mat[i])==0:
        for t in range(7):
            j = random.randint(0,num_of_nodes)
            if i != j: adj_mat[i][j], adj_mat[j][i] = 1, 1
    if np.sum(adj_mat[i])==1:
        for t in range(2):

            j = random.randint(0, num_of_nodes)
            if i != j: adj_mat[i][j], adj_mat[j][i] = 1, 1

deg = np.sum(adj_mat,axis=0)
if np.any(deg==0): print('Exist degree is 0! {}'.format(np.where(deg==0)))

np.save(dirs+'/{}_label'.format(data),true_labels)
np.save(dirs+'/{}_adj'.format(data),adj_mat)
np.save(dirs+'/{}_feat'.format(data),node_features)
print("@END(Graph-V{}) nodes:{}  edges:{}  degree1:{}  degree2:{} ".format(data, G.number_of_nodes(), G.size(), np.sum(adj_mat[:nodes_num_li[0],:nodes_num_li[0]])/2,np.sum(adj_mat[nodes_num_li[0]:,nodes_num_li[0]:])/2,))
a1 = np.sum(adj_mat[:1000, :1000])/1000
a2 = np.sum(adj_mat[1000:, 1000:])/ 1000
a3 = np.sum(adj_mat)/ 1000
print("d1:{}   d2:{}   d3:{}   ratio:{}".format(a1, a2, (a3 - a1 - a2)/2, (a3 - a1 - a2) / (2*a1)))