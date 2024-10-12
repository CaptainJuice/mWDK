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

#
# import math
#
# # 正十边形的半径
# radius = 1
#
# # 正十边形的中心点坐标
# center_x = 0
# center_y = 0
# node_features = []
# # 生成每个点的坐标并直接打印出来
# for i in range(10):
#     angle = math.radians(36 * i)  # 将角度转换为弧度
#     x = center_x + radius * math.cos(angle)+random.randint(1,100) *0.01
#     y = center_y + radius * math.sin(angle)+random.randint(1,100) *0.01
#     node_features.append([x,y])
# node_features = np.array(node_features)
# data = 'reg'
data = 'inituition'

dirs = "E:/Graph Clustering/dataset/artificial data/raw_data/{}".format(data)
if not os.path.exists(dirs):
    os.makedirs(dirs)
# print("Genarating ···".format(data))
#
# G = nx.random_graphs.random_regular_graph(8,10)  #生成包含10个节点、每个节点有2个邻居的规则图RG
# adj_mat = nx.adjacency_matrix(G).todense()

# node_features = Gaussian_Distribution(2,10,0,1)
# node_features = np.load(dirs+'/{}_feat.npy'.format(data))
adj_mat = [[0, 1, 0, 1, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 1, 0],
           [1, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 1, 0, 1, 0],
           [0, 0, 1, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 1, 0, 1, 0],]

node_features = np.array([[1,1],
                 [1,4],
                 [4,4],
                 [4,1],
                 [2,2],
                 [2,3],
                 [3,3],
                 [3,2],])

true_labels = np.array([1]*8)
plt.scatter(node_features[:, 0], node_features[:, 1],s=5, c=true_labels, cmap="rainbow")
plt.title('raw:Graph-{}'.format(data))
plt.savefig(dirs+'/raw.png')
plt.show()


np.save(dirs+'/{}_label'.format(data),true_labels)
np.save(dirs+'/{}_adj'.format(data),adj_mat)
np.save(dirs+'/{}_feat'.format(data),node_features)

#

pos = node_features



num_nodes = 10

# # 每个节点的度数
# for d in range(2,10,2):
#     degree = d
#
#     # 生成正则图
#     graph = nx.random_regular_graph(degree, num_nodes)
#
#     # 绘制图形
#     pos = nx.circular_layout(graph)
#     nx.draw(graph, pos, with_labels=False, node_color='red', node_size=300, edge_color='black', linewidths=1)
#
#     # 显示图形
#     plt.title("Regular Graph")
#     plt.axis('off')
#     plt.show()