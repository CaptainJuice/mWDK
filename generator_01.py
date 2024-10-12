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
    mean = np.zeros(dim)+mu
    cov = np.eye(dim) * sigma

    # nums = [i for i in range(20)]
    # weights = [0.1*random.randint(2,6) for i in range(20)]
    # for i in range(cov.shape[0]):
    #     for j in range(i,cov.shape[0]):
    #         cov[i][j] = random.choices(nums,weights=weights,k=1)[0]
    #         cov[j][i] = cov[i][j]

    data = np.random.multivariate_normal(mean, cov, num_nodes)

    return data
data = 'time'
dirs = "E:/Graph Clustering/dataset/artificial data/raw_data/Graph_{}".format(data)
if not os.path.exists(dirs):
    os.makedirs(dirs)
#################################################################################################################
dim = 100
nodes_num_li = [50000,50000]
average_degree = 2
#################################################################################################################

num_of_class = len(nodes_num_li)
num_of_nodes = np.sum(nodes_num_li)
true_labels = [0 if i<nodes_num_li[0] else 1 for i in range(num_of_nodes)]
node_features=[]
mu_li = [0,1.5]
sigma_li = [55,55]
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
# tsne = TSNE(n_components=2,perplexity=35,learning_rate=0.001,init='pca',n_iter=4000)
# node_features_tsne = tsne.fit_transform(node_features)
# plt.scatter(node_features[:,0], node_features[:,1],s=4, c=true_labels, cmap="rainbow")
# plt.show()
#
# node_features = preprocessing.normalize(node_features,'l2',axis=0)
# min_max_scaler = preprocessing.MinMaxScaler()
# node_features = min_max_scaler.fit_transform(node_features)
# plt.scatter(node_features[:,0], node_features[:,1],s=4, c=true_labels, cmap="rainbow")

# plt.scatter(node_features_tsne[:,0], node_features_tsne[:,1],s=5, c=true_labels, cmap="rainbow")
# plt.title('Visualization of Graph-{}(t-SNE)'.format(data))
# plt.savefig(dirs+'/tsne.png')
# plt.show()
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
        tar = [i for i in range(-average_degree + 3, average_degree * 2)]
        tar += [i for i in range(average_degree * 2, average_degree * 40,5)]
        tar =np.array(tar)
        prob = np.array([max(1 - 0.15 * i * np.sqrt(i), 0.02) for i in range(len(tar))])

        prob[-6:] = prob[-6:]*2
        prob = prob / np.sum(prob)
        rand = np.random.choice(a=tar, size=1, replace=True, p=prob)[0]

        p = average_degree + rand
        q=j
        if j>=nodes_num_li[0]:
            q=j-nodes_num_li[0]

        k_ind = heapq.nsmallest(p, range(len(dis[q])), dis[q].take)
        k_ind =[cluster[i] for i in k_ind]

        for k in k_ind:

            if j != k: adj_mat[j][k], adj_mat[k][j] = 1, 1
deg = np.sum(adj_mat,axis=0)

adj_plot(adj_mat,true_labels,100)
cnt, cmt = 0, 0
for i in range(node_features.shape[0]):
    if deg[i]==1:
        p = 0
        s = 0
    elif deg[i]==2:
        p = random.randint(1, 400)  ####70 50 50 40
        s = random.randint(1, 200)
    elif deg[i]<np.mean(deg)+3:
        p = random.randint(1, 100)
        s = random.randint(1, 50)
    else:
        p = random.randint(1, 10)
        s = random.randint(1, 6)

    for j in range(node_features.shape[0]):
        if (i < nodes_num_li[0] and j > nodes_num_li[0]) or (i > nodes_num_li[0] and j < nodes_num_li[0]):
            if s == 1:
                t = random.randint(1, 300)#100 90
                if t == 1:
                    adj_mat[i][j], adj_mat[j][i] = 1, 1

            elif p==1:
                t = random.randint(1, 200)
                if t == 1:
                    adj_mat[i][j], adj_mat[j][i] = 1, 1


        else:
            inner = random.randint(1, 1500)



            if adj_mat[i][j] == 1 and inner > 500 and deg[i]>2 and deg[j]>1:
                cnt +=1
                adj_mat[i][j], adj_mat[j][i] = 0, 0
                deg[i]-=1
                deg[j]-=1

            elif adj_mat[i][j] == 0 and inner <5 and i!=j:
                cmt+=1
                adj_mat[i][j], adj_mat[j][i] = 1, 1
                deg[i]+=1
                deg[j]+=1
print("del:{} add:{}".format(cnt,cmt))
#################################################################################
for j in range(node_features.shape[0]):
    if deg[j]==1:
        p = 0
        s = 0
    elif deg[j]==2:
        p = random.randint(1, 500)  ####70 50 50 40
        s = random.randint(1, 300)
    elif deg[j]<np.mean(deg)+3:
        p = random.randint(1, 200)
        s = random.randint(1, 160)
    else:
        p = random.randint(1, 10)
        s = random.randint(1, 6)

    for i in range(node_features.shape[0]):
        if (j < nodes_num_li[0] and i > nodes_num_li[0]) or (j > nodes_num_li[0] and i < nodes_num_li[0]):
            if s == 1:
                t = random.randint(1, 300)#100 90
                if t == 1:
                    adj_mat[i][j], adj_mat[j][i] = 1, 1

            elif p==1:
                t = random.randint(1, 200)
                if t == 1:
                    adj_mat[i][j], adj_mat[j][i] = 1, 1

for i in range(nodes_num_li[0],adj_mat.shape[0]):
    for j in range(i,adj_mat.shape[0]):
        if adj_mat[i][j]==1:
            s = random.randint(1,11)
            if s >= 7:
                adj_mat[i][j],adj_mat[j][i]=0,0

adj_plot(adj_mat,true_labels,100)
# nx.draw(G,node_color=true_labels,node_size=1.5,alpha=0.15)
# nx.draw_networkx_nodes(G, node_features, node_size=1, node_color=true_labels)  # 画节点
# nx.draw_networkx_edges(G, node_features, alpha=0.3, width=1)  # 画边
# nx.draw_networkx_labels(G, X, node_labels=label,font_size=20)
# nx.draw_networkx_nodes(G, npos, node_size=5, node_color=label)  # 绘制节点
# nx.draw_networkx_edges(G, npos, adj_mat)  # 绘制边
# nx.draw_networkx_labels(G, npos, nlabels)  # 标签
plt.show()
########################### check #################################
for i in range(adj_mat.shape[0]):
    if np.sum(adj_mat[i])==0:
        for t in range(7):
            j = random.randint(0,num_of_nodes-1)
            if i != j: adj_mat[i][j], adj_mat[j][i] = 1, 1
    if np.sum(adj_mat[i])==1:
        for t in range(2):

            j = random.randint(0, num_of_nodes-1)
            if i != j: adj_mat[i][j], adj_mat[j][i] = 1, 1

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
plt.rcParams['axes.facecolor']='snow'

# GG = nx.random_partition_graph(nodes_num_li,0.0041,0.0007)
# adj_mat= np.array(nx.adjacency_matrix(GG).todense())
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
#
# for i in range(adj_mat.shape[0]):
#     if np.sum(adj_mat[i])==0:
#         for p in range(6):
#             t = random.randint(1,2000)
#             if t != i:
#                 adj_mat[i][t], adj_mat[t][i] = 1, 1
# deg = np.sum(adj_mat,axis=0)

if np.any(deg==0): print('Exist degree is 0! {}'.format(np.where(deg==0)))

np.save(dirs+'/Graph_{}_label'.format(data),true_labels)
np.save(dirs+'/Graph_{}_adj'.format(data),adj_mat)
np.save(dirs+'/Graph_{}_feat'.format(data),node_features)
print("@END(Graph-V{}) nodes:{}  edges:{}  degree1:{}  degree2:{} ".format(data, G.number_of_nodes(), G.size(), np.sum(adj_mat[:nodes_num_li[0],:nodes_num_li[0]])/2,np.sum(adj_mat[nodes_num_li[0]:,nodes_num_li[0]:])/2,))
a1 = np.sum(adj_mat[:1000, :1000])/1000
a2 = np.sum(adj_mat[1000:, 1000:])/ 1000
a3 = np.sum(adj_mat)/ 1000

print("d1:{}   d2:{}   d3:{}   ratio:{}".format(a1, a2, (a3 - a1 - a2)/2, (a3 - a1 - a2) / (2*a1)))