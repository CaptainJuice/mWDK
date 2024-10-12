import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
from utils import IK_fm_dot
import random
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

data = 2
dirs = "E:/Graph Clustering/dataset/artificial data/raw_data/Graph_{}".format(data)
if not os.path.exists(dirs):
    os.makedirs(dirs)
print("Genarating ···".format(data))

#################################################################################################################

dim = 2
nodes_num_li = [500,500]
mu_li = [0,4]
sigma_li = [8,8]
rate1 = 0.005
rate2 = 0.0006
num_of_class = len(nodes_num_li)
num_of_nodes = np.sum(nodes_num_li)
X = np.zeros((num_of_nodes,dim))
true_labels = []
for id in range(len(nodes_num_li)):
    nodes_num = nodes_num_li[id]
    label = [id] * nodes_num
    true_labels.extend(label)

f1 = nodes_num_li[0]
f2 = nodes_num_li[1]
f = f1+f2
for d in range(dim):
    if d<500:
        for i in range(f):
            if i <f1:
                rd = random.randint(1,100)
                if rd == 1:
                    X[i][d] = 1
                else:
                    X[i][d] = 0
            else:
                if i < f1:
                    rd = random.randint(1, 110)
                    if rd == 1:
                        X[i][d] = 1
                    else:
                        X[i][d] = 0
    else:

        for i in range(f):
            if i <f1:
                rd = random.randint(1,110)
                if rd == 1:
                    X[i][d] = 1
                else:
                    X[i][d] = 0
            else:

                rd = random.randint(1, 100)
                if rd == 1:
                    X[i][d] = 1
                else:
                    X[i][d] = 0



G1 = nx.random_partition_graph(nodes_num_li, rate1, rate2)

adj_mat1 = np.array(nx.adjacency_matrix (G1).todense().tolist())


adj_mat = np.zeros_like(adj_mat1)
G2 = nx.random_partition_graph(nodes_num_li, 0.01, rate2)
adj_mat2 = np.array(nx.adjacency_matrix (G2).todense().tolist())
for i in range(adj_mat1.shape[0]):
    for j in range(adj_mat1.shape[1]):
        if i < nodes_num_li[0] and j < nodes_num_li[0]:
            adj_mat[i][j] = adj_mat1[i][j]
        else:
            adj_mat[i][j] = adj_mat2[i][j]
s = np.sum(adj_mat)
s1 = np.sum(adj_mat[:nodes_num_li[0],:nodes_num_li[0]])

s2 = np.sum(adj_mat[nodes_num_li[0]:,nodes_num_li[0]:])
print(s1,s2,s,s-s1-s2)
node_features =np.array(X)
true_labels = np.array(true_labels)
#
from sklearn import preprocessing
tsne = TSNE(n_components=2,perplexity=25,n_iter=1000,learning_rate=250)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1],s=5, c=true_labels, cmap="rainbow")
plt.title('Visualization of Graph-{}(t-SNE)'.format(data))
plt.savefig(dirs+'/tsne.png')
plt.show()
print("@END(Graph-V{}) nodes:{}  edges:{} ".format(data, G1.number_of_nodes(), G1.size()))



np.save(dirs+'/graph_{}_label'.format(data),true_labels)
np.save(dirs+'/graph_{}_adj'.format(data),adj_mat)
np.save(dirs+'/graph_{}_feat'.format(data),node_features)

pos = X_tsne
G =nx.from_numpy_matrix(adj_mat)
nx.draw_networkx_nodes(G, pos, node_size=2, node_color=true_labels)  # 画节点
nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.3)  # 画边
# node_labels = nx.get_node_attributes(G, 'name')
# nx.draw_networkx_labels(G, X, node_labels=label,font_size=20)
# nx.draw_networkx_nodes(G, npos, node_size=5, node_color=label)  # 绘制节点
# nx.draw_networkx_edges(G, npos, adj_mat)  # 绘制边
# nx.draw_networkx_labels(G, npos, nlabels)  # 标签
plt.show()


from sklearn.cluster import SpectralClustering,KMeans
from sklearn.metrics import accuracy_score,normalized_mutual_info_score

X = preprocessing.normalize(X,'l2')

sim = X.dot(X.T)
spectral = SpectralClustering(n_clusters=num_of_class, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize', random_state=66)
spectral.fit(sim)
predict_labels = spectral.fit_predict(sim)
nmi = normalized_mutual_info_score(true_labels,predict_labels)
print(nmi)