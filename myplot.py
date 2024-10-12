import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy import interpolate

from utils import WL, WL_weighted_con, load_data, WL_noconcate, IGK_WL_noconcate, IK_fm_dot, WL_noconcate_gcn, pplot2, sub_wl, WL_test, WL_noconcate_one
from utils import create_adj_avg_gcn,create_adj_avg
import numpy as np
from sklearn.manifold import TSNE
import clustering_methods as cmd
from sklearn import preprocessing
import math
from scipy.spatial.distance import pdist, squareform
import random
import random
import copy
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import scipy.sparse as sp
import torch
import torch.nn.functional as F

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
def IK_inne_fm(X, psi, t=100):
    onepoint_matrix = np.zeros((X.shape[0], (int)(t * psi)), dtype=int)
    for time in range(t):
        sample_num = psi  #
        sample_list = [p for p in range(len(X))]
        sample_list = random.sample(sample_list, sample_num)
        sample = X[sample_list, :]

        tem1 = np.dot(np.square(X), np.ones(sample.T.shape))  # n*psi
        tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)  # n*psi

        # tem = np.dot(np.square(sample), np.ones(sample.T.shape))
        # sample2sample = tem + tem.T - 2 * np.dot(sample, sample.T)
        sample2sample = point2sample[sample_list, :]
        row, col = np.diag_indices_from(sample2sample)
        sample2sample[row, col] = 99999999
        radius_list = np.min(sample2sample, axis=1)  # 每行的最小值形成一个行向量

        min_point2sample_index = np.argmin(point2sample, axis=1)
        min_dist_point2sample = min_point2sample_index + time * psi
        point2sample_value = point2sample[range(len(onepoint_matrix)), min_point2sample_index]
        ind = point2sample_value < radius_list[min_point2sample_index]
        onepoint_matrix[ind, min_dist_point2sample[ind]] = 1
    return onepoint_matrix
def IDK(X, psi, t=200):
    # point_fm_list=IK_inne_fm(X=X,psi=psi,t=t)
    point_fm_list = IK_fm_dot(X=X, psi=psi, t=t)
    feature_mean_map = np.mean(point_fm_list, axis=0)
    idk_score = np.dot(point_fm_list, feature_mean_map) / t
    # idk_score = (idk_score - np.min(idk_score)) / (np.max(idk_score) - np.min(idk_score))
    # idk_score = (idk_score - np.mean(idk_score)) / np.std(idk_score)
    # idk_score = (idk_score - np.min(idk_score)) / (np.max(idk_score) - np.min(idk_score))
    idk_score = idk_score / np.max(idk_score)

    return idk_score
def neighborhood_overlap(G, u, v):
    u_n = G.degree()[u]
    v_n = G.degree()[v]
    C_n = len(list(nx.common_neighbors(G, u, v)))
    if u_n - 1 + v_n - 1 - C_n == 0:
        return 0
    else:
        return abs(C_n / (u_n - 1 + v_n - 1 - C_n))
def get_neigbors(g, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1, depth + 1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
    return output
def WWL(node_features, adj, h):
    emb = np.zeros_like(node_features)
    deg = np.sum(adj, axis=1)
    G = nx.from_numpy_matrix(adj_mat)
    neighbors_list = [get_neigbors(G, tar, depth=1)[1] for tar in range(node_features.shape[0])]
    for i in range(node_features.shape[0]):
        r = adj[i][i] / deg[i]
        p_change = np.array([r, 1 - r])
        is_change = np.random.choice([0, 1], p=p_change.ravel())
        # is_change=1

        if is_change == 0:
            # print(i)

            emb[i] = node_features[i]
        else:
            neighbor = neighbors_list[i]
            deg_neighbor = deg[neighbor]
            p_concat = deg_neighbor / np.sum(deg_neighbor)
            if deg_neighbor[0] == np.max(deg_neighbor):

                emb_temp = node_features[i]
            else:
                emb_temp = node_features[i] * r

            for id in range(len(neighbor)):
                emb_temp += p_concat[id] * node_features[neighbor[id]]
            emb[i] = emb_temp

    #     deg = 1 / deg
    # deg_mat = np.diag(deg)
    #
    # adj = adj.dot(deg_mat).T  # +adj_cur.dot(deg_mat.T).T
    #
    # np.fill_diagonal(adj, 5)
    # power = np.linalg.matrix_power(adj, h)
    # embedding = math.pow(0.5,h) * (np.dot(power, node_features))
    return emb
def CW(node_features, adj_mat, h):
    D = np.identity(node_features.shape[0])
    for i in range(h):
        s = np.max(D, axis=0)
        for j in range(D.shape[0]):
            for k in range(D.shape[1]):
                if D[i][k] < s[j]:
                    D[i][k] = 0
                else:
                    D[i][k] = 1
                    # s[j]=100000
        D = D.dot(adj_mat)
    return D
# 抽取txt中的数据
def read_txt(data):
    g = nx.read_weighted_edgelist(data)
    print(g.edges())
    return g
def gDegree(G):
    """
    将G.degree()的返回值变为字典
    """
    node_degrees_dict = {}
    for i in G.degree():
        node_degrees_dict[i[0]] = i[1]
    return node_degrees_dict.copy()
def kshell(G):
    graph = G.copy()
    importance_dict = {}
    level = 1
    while len(graph.degree):
        importance_dict[level] = []
        while True:
            level_node_list = []
            for item in graph.degree:
                if item[1] <= level:
                    level_node_list.append(item[0])
            graph.remove_nodes_from(level_node_list)
            importance_dict[level].extend(level_node_list)
            if not len(graph.degree):
                return importance_dict
            if min(graph.degree, key=lambda x: x[1])[1] > level:
                break
        level = min(graph.degree, key=lambda x: x[1])[1]
    return importance_dict
def sumD(G):
    """
    计算G中度的和
    """
    G_degrees = gDegree(G)
    sum = 0
    for v in G_degrees.values():
        sum += v
    return sum


def getNodeImportIndex(G):
    """
    计算节点的重要性指数
    """
    sum = sumD(G)
    I = {}
    G_degrees = gDegree(G)
    for k, v in G_degrees.items():
        I[k] = v / sum
    return I


def Entropy(G):
    """
    Entropy(G) 计算出G中所有节点的熵
    I 为重要性
    e 为节点的熵sum += I[i]*math.log(I[i])
    """
    I = getNodeImportIndex(G)
    e = {}
    for k, v in I.items():
        sum = 0
        for i in G.neighbors(k):
            sum += I[i] * math.log(I[i])
        sum = -sum
        e[k] = sum
    return e


def kshellEntropy(G):
    """
    kshellEntropy(G) 是计算所有壳层下，所有节点的熵值
    例：
    {28: {'1430': 0.3787255719932099,
          '646': 0.3754626894107377,
          '1431': 0.3787255719932099,
          '1432': 0.3787255719932099,
          '1433': 0.3754626894107377
          ....
    ks is a dict 显示每个壳中的节点
    e 计算了算有节点的熵
    """
    ks = kshell(G)
    e = Entropy(G)
    ksES = {}
    ksIs = sorted(ks.keys(), reverse=True)
    for ksI in ksIs:
        ksE = {}
        for i in ks[ksI]:
            ksE[i] = e[i]
        ksES[ksI] = ksE
    return ksES


def kshellEntropySort(G):
    ksE = kshellEntropy(G)
    ksES = []
    ksIs = sorted(ksE.keys(), reverse=True)
    for ksI in ksIs:
        t = sorted([(v, k) for k, v in ksE[ksI].items()], reverse=True)

        # 把熵值一样的节点放在一个集合中
        t_new = {}
        for i in t:
            t_new.setdefault(i[0], list()).append(i[1])
        # 按熵值排序变成列表
        t = sorted([(k, v) for k, v in t_new.items()], reverse=True)

        # 把相同熵值的节点列表打乱顺序，相当于随机选择
        sub_ksES = []
        for i in t:
            if len(i[1]) == 1:
                sub_ksES += i[1]
            else:
                random.shuffle(i[1])
                sub_ksES += i[1]

        ksES.append(sub_ksES)
    #         ksES.append(list(i[1] for i in t))
    return ksES


def getRank(G):
    rank = []
    rankEntropy = kshellEntropySort(G)
    while (len(rankEntropy) != 0):
        for i in range(len(rankEntropy)):
            rank.append(rankEntropy[i].pop(0))
        while True:
            if [] in rankEntropy:
                rankEntropy.remove([])
            else:
                break
    return rank


def get_neighbors_single(g, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1, depth + 1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
    return output


def get_neighbors_all(G, node, depth=1):
    temp = [node]
    single_neighbors = get_neighbors_single(G, node, depth)
    for i in range(depth):
        temp.extend(single_neighbors[i + 1])
    return np.array(temp)


def WL_my(node_features, adj_mat, h, deg):
    deg2 = np.sum(adj_mat, axis=0)

    deg2 = 1 / deg2
    # deg = 1 / deg
    deg = np.sqrt(deg)
    deg2 = np.sqrt(deg2)

    deg_mat1 = np.diag(deg)
    deg_mat2 = np.diag(deg2)

    adj_mat = adj_mat.dot(deg_mat1).T  # +adj_cur.dot(deg_mat.T).T
    adj_mat = adj_mat.dot(deg_mat2)

    np.fill_diagonal(adj_mat, 1)
    power = np.linalg.matrix_power(adj_mat, h)

    embedding = math.pow(0.5, h) * (np.dot(power, node_features))
    return embedding


def mywp(G, adj_mat):
    go_time = 1000
    max_length = 100
    res = np.zeros_like(adj_mat)
    for i in range(adj_mat.shape[0]):
        # print("@node={} gotime={}/{}".format(i, go_time, go_time))
        for rep in range(go_time):
            current = i
            temp = np.zeros_like(adj_mat[0])
            for j in range(max_length):
                neighbors = get_neighbors_single(G, current, depth=1)[1]
                next = random.sample(neighbors, 1)[0]
                temp[next] += 1
                current = next

    return res / (go_time * max_length)


def my_recall(G, adj_mat):
    go_time = 1000
    max_length = 25
    r = [i for i in range(adj_mat.shape[0])]
    res = np.zeros_like(adj_mat)
    li = random.sample(r, 200)
    for i in li:
        # print("@node={} gotime={}/{}".format(i, go_time, go_time))
        tar = i
        neighbors_of_tar = get_neighbors_single(G, tar, depth=1)[1]
        for j in neighbors_of_tar:
            cnt = 0
            current = j
            for rep in range(go_time):
                for t in range(max_length):

                    neighbors_of_neighbors = get_neighbors_single(G, current, depth=1)[1]
                    next = random.sample(neighbors_of_neighbors, 1)[0]
                    current = next
                    if next == tar:
                        cnt += 1
                        current = j
                        continue
            res[i][j] = cnt

    return res / (go_time * max_length)


def drop_edge(adj_mat):
    deg = np.sum(adj_mat, axis=0)
    r = random.randint(1, 3)
    if r != 1:
        return adj_mat
    cnt = 0
    cmt = 0

    adj = np.zeros_like(adj_mat)
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] == 1:
                r = random.randint(1, 5)
                if r == 1:
                    adj[i][j] = 0
                    cnt += 1
            else:
                r = random.randint(1, deg[j] * 15)
                if r == 1:
                    adj[i][j] = 1
                    cmt += 1
    print(cnt, cmt)

    return adj


def COP_K_means(X, n_clusters=3, Con1=None, Con2=None):
    clusters = np.random.choice(len(X), n_clusters)
    clusters = X[clusters]
    labels = np.array([-1 for i in range(len(X))])

    def validata_constrained(d, c, Con1, Con2):
        for dm, value in enumerate(Con1[d]):  # should in the same group
            if value == 0:
                continue
            if labels[dm] == -1 or labels[dm] == c:  # has not allocated or ...
                continue
            if labels[dm] != -1 and labels[dm] != c:  # has allocated
                return False

        for dm, value in enumerate(Con2[d]):  # cannot in the same group
            if value == 0:
                continue
            if labels[dm] == -1 or labels[dm] != c:  # has not allocated or ...
                continue
            if labels[dm] != -1 and labels[dm] == c:  # has allocated
                return False

        return True

    while True:
        labels_new = np.array([-1 for i in range(len(X))])
        for i, xi in enumerate(X):
            close_list = np.argsort([np.linalg.norm(xi - cj) for cj in clusters])

            unexpect = True
            for index in close_list:
                if validata_constrained(i, index, Con1, Con2):
                    unexpect = False
                    labels_new[i] = index
                    break
            if unexpect:
                raise Exception("Can not utilize COP-k-Means algorithm inside the dataset.")

        if sum(labels != labels_new) == 0:
            break

        for j in range(n_clusters):
            clusters[j] = np.mean(X[np.where(labels_new == j)], axis=0)
        labels = labels_new.copy()
    return labels


def wl_idk(G, node_features, adj_mat, idk, h):
    adj_mat = adj_mat.astype(float)

    for i in range(adj_mat.shape[0]):

        neighbors = get_neighbors_all(G, i, depth=1)
        idk_neighbors = [idk[k] for k in neighbors]

        for j in neighbors:
            # neighbors = get_neighbors_all(G,i,depth=1)
            if idk[j] < idk[i]:
                adj_mat[i][j] = 0
        if idk[i] > 0.9:
            adj_mat[i] = 0

        # ind =idk_neighbors.index(min(idk_neighbors))
        # adj_mat[i][neighbors[ind]] =0.5
        # ind = idk_neighbors.index(max(idk_neighbors))
        # adj_mat[i][neighbors[ind]] = 2

    deg = np.sum(adj_mat, axis=1)
    deg[deg == 0] = 1
    deg = 1 / deg
    deg_mat = np.diag(deg)

    adj_mat = adj_mat.dot(deg_mat).T  # +adj_mat.dot(deg_mat).T
    # adj_mat =preprocessing.normalize(adj_mat,"l2")

    np.fill_diagonal(adj_mat, 1)
    power = np.linalg.matrix_power(adj_mat, h)

    embedding = math.pow(0.5, h) * (np.dot(power, node_features))
    return embedding


def wl_idk2(G, node_features, adj_mat, idk, h):
    adj_mat = adj_mat.astype(float)
    adj = np.eye(node_features.shape[0])
    np.fill_diagonal(adj_mat, 1)
    aaa = adj_mat.copy()
    for t in range(h):
        for i in range(adj_mat.shape[0]):

            neighbors = get_neighbors_all(G, i, depth=1)
            idk_neighbors = [idk[k] for k in neighbors]
            # m =np.median(idk)
            for j in neighbors:
                # neighbors = get_neighbors_all(G,i,depth=1)
                if idk[j] > idk[i]:
                    adj_mat[i][j] = 0

            # ind =idk_neighbors.index(min(idk_neighbors))
            # adj_mat[i][neighbors[ind]] =0.5

        deg = np.sum(adj_mat, axis=1)
        deg[deg == 0] = 1
        deg = 1 / deg
        deg_mat = np.diag(deg)

        adj_mat = adj_mat.dot(deg_mat).T  # +adj_mat.dot(deg_mat).T
        # adj_mat =preprocessing.normalize(adj_mat,"l2")

        adj = adj.dot(adj_mat)
        embedding = WL(node_features, aaa, t)
        idk = IDK(embedding, 32, 200)
        adj_mat = aaa.copy()

    power = adj

    embedding = math.pow(0.5, h) * (np.dot(power, node_features))
    return embedding


import heapq


def getListMaxNumIndex(num_list, topk=3):
    num_list = num_list.tolist()
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    max_num_index = map(num_list.index, heapq.nlargest(topk, num_list))
    min_num_index = map(num_list.index, heapq.nsmallest(topk, num_list))
    max_num_index = list(max_num_index)
    min_num_index = list(min_num_index)
    return max_num_index

def WL_noconcate_zero(G,node_features, adj_mat,h):
    new_adj = create_adj_avg(adj_mat)
    adj = new_adj.copy()


    embedding = np.zeros_like(node_features,dtype=float)

    for i in range(adj_mat.shape[0]):
        adj[i] = 0
        adj[:,i] = 0
        power = np.linalg.matrix_power(adj, h)
        embeddings = math.pow(0.5, h) * (np.dot(power, node_features))
        nei = get_neighbors_all(G,i,1)[1:]
        embs = embeddings[nei]
        embedding[i] = np.mean(embs,axis=0)
    # np.fill_diagonal(power, 0)
    return embedding
##
def plot_graph(G,emb,label,sample_list):

    tsne = TSNE(n_components=2, perplexity=50, n_iter=1000)
    emb_tsne = tsne.fit_transform(emb)
    mapping = {}

    for i, j in enumerate(sample_list):
        mapping[j] = i
    G = nx.relabel_nodes(G, mapping)

    nx.draw(G,emb_tsne,node_color=label,node_size=15)
    plt.show()


def IK_fm_graph(G,adj_mat,X,psi,t,h):
    # ### way1
    # adj=adj_mat.dot(adj_mat)
    # adj = np.where(adj>0,1,0)
    # adj2 = adj_mat.dot(adj_mat).dot(adj_mat)
    # adj = adj_mat+adj
    #
    # X1 =WL_noconcate(X,adj_mat,6)

    # X =WL_noconcate(X,adj_mat,1)

    ### way2


    onepoint_matrix = np.zeros((X.shape[0], (int)(t * psi)), dtype=int)
    x_index=np.arange(len(X))
    for time in range(t):
        sample_num = psi  #
        sample_list = [p for p in range(len(X))]  # [0, 1, 2, 3]
        sample_list = random.sample(sample_list, sample_num)  # [1, 2]
        sample = X[sample_list, :]  # array([[ 4,  5,  6,  7], [ 8,  9, 10, 11]])
        #sim
        point2sample =np.dot(X,sample.T)
        min_dist_point2sample = np.argmax(point2sample, axis=1)+time*psi
        onepoint_matrix[x_index,min_dist_point2sample]=1

    return onepoint_matrix



def mad_value_source(in_arr, mask_arr, distance_metric='cosine', digt_num=4, target_idx=None):
    dist_arr = pairwise_distances(in_arr, in_arr, metric=distance_metric)

    mask_dist = np.multiply(dist_arr, mask_arr)

    divide_arr = (mask_dist != 0).sum(1) + 1e-8

    node_dist = mask_dist.sum(1) / divide_arr

    if target_idx.any() == None:
        mad = np.mean(node_dist)
    else:
        node_dist = np.multiply(node_dist, target_idx)
        mad = node_dist.sum() / ((node_dist != 0).sum() + 1e-8)

    mad = round(mad, digt_num)

    return mad

def mad_value(in_arr, mask_arr, distance_metric='cosine', digt_num=4, target_idx=None):
    dist_arr = pairwise_distances(in_arr, in_arr, metric=distance_metric)

    mask_dist = np.multiply(dist_arr, mask_arr)

    divide_arr = (mask_dist != 0).sum(1) + 1e-8

    node_dist = mask_dist.sum(1) / divide_arr


    mad = np.mean(node_dist)

    mad = round(mad, digt_num)

    return mad


def mad_value_entropy(in_arr, mask_arr, distance_metric='cosine', digt_num=4, target_idx=None):
    mean_map = np.sum(in_arr,axis=0)
    sim =in_arr.dot(mean_map.T)
    pp = sim/np.sum(sim)
    mad = 0
    for i in pp:
        mad-=i*np.log(i)





    return mad
import heapq
import heapq


def mad_gap_regularizer(embedding, neb_mask, rmt_mask, target_idx):

    simi = embedding.dot(embedding.T)
    dist = 1 - simi

    neb_dist = np.dot(dist, neb_mask)
    rmt_dist = np.dot(dist, rmt_mask)

    divide_neb = np.sum(neb_dist)
    divide_rmt = np.sum(rmt_dist)

    neb_mean_list = neb_dist.sum(1) / divide_neb
    rmt_mean_list = rmt_dist.sum(1) / divide_rmt

    neb_mad = np.mean(neb_mean_list[target_idx])
    rmt_mad = np.mean(rmt_mean_list[target_idx])

    mad_gap = rmt_mad - neb_mad

    return mad_gap
from scipy.ndimage import gaussian_filter1d
def plot_twin(_y1, _y2,_y3, _y4,_y5,_y6,_y7,_y8, _ylabel1, _ylabel2):
    h=np.array([i for i in range(len(_y1))])


    # _y1 = gaussian_filter1d(_y1, sigma=5)
    # _y2 = gaussian_filter1d(_y2, sigma=5)
    # _y3 = gaussian_filter1d(_y3, sigma=5)

    # _y5= gaussian_filter1d(_y5, sigma=5)
    # _y6= gaussian_filter1d(_y6, sigma=5)
    # _y7= gaussian_filter1d(_y7, sigma=5)
    # _y8 = gaussian_filter1d(_y8, sigma=5)

    func1 = interpolate.interp1d(h, _y1, kind='cubic')
    func2 = interpolate.interp1d(h, _y2, kind='cubic')
    func3 = interpolate.interp1d(h, _y3, kind='cubic')
    func4 = interpolate.interp1d(h, _y4, kind='cubic')
    func5 = interpolate.interp1d(h, _y5, kind='cubic')
    func6 = interpolate.interp1d(h, _y6, kind='cubic')
    func7 = interpolate.interp1d(h, _y7, kind='cubic')
    func8 = interpolate.interp1d(h, _y8, kind='cubic')
    newh = np.arange(np.min(h), np.max(h), 0.1)

    # 利用xnew和func函数生成ynew,xnew数量等于ynew数量
    _y1 = func1(newh)
    _y2 = func2(newh)
    _y3 = func3(newh)
    _y4 = func4(newh)
    _y5 = func5(newh)
    _y6 = func6(newh)
    _y7 = func7(newh)
    _y8 = func8(newh)




    fig, ax1 = plt.subplots(dpi=300)

    color = 'blue'
    ax1.set_xlabel('Iterations (h)')
    ax1.set_ylabel(_ylabel1, color=color)
    ax1.plot(h,_y1, color=color, marker='o',markersize=3,label='MAD_WL')
    ax1.plot(h,_y2, color=color, marker='x',markersize=4,label='MAD_IKWL')
    ax1.plot(h,_y3, color=color, marker='^',markersize=3, label='MAD_IIKWL')
    ax1.plot(h,_y4, color=color, marker='s',markersize=3, label='MAD_GKWL')

    ax1.tick_params(axis='y', labelcolor=color,labelsize=8)


    ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴

    color = '#FA7F6F'
    ax2.set_ylabel(_ylabel2, color=color)
    ax2.plot(h,_y5, color=color, marker='o',markersize=3,label='NMI_WL')
    ax2.plot(h,_y6, color=color, marker='x',markersize=4,label='NMI_IKWL')
    ax2.plot(h,_y7, color=color, marker='^', markersize=3,label='NMI_IIKWL')
    ax2.plot(h, _y8, color=color, marker='s', markersize=3, label='NMI_GKWL')


    ax2.tick_params(axis='y', labelcolor=color,labelsize=8)
    fig.legend(bbox_to_anchor=(1.11, 1), loc=2, borderaxespad=0, bbox_transform=ax1.transAxes,prop = {'size':6},framealpha=1)
    fig.tight_layout()
    # plt.legend()  # 让图例生效
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.8, left=0.1, hspace=0, wspace=0)
    plt.margins(0.1, 0.1)
    plt.xticks(np.arange(0, len(_y1),5), [i for i in range(0,len(_y1),5)])
    plt.show()

def sim_plot(y1,y2,y3,y4,y5,y6,name):
    x = [i for i in range(h+1)]
    fig, ax = plt.subplots()  # 创建图实例
    # y1 = gaussian_filter1d(y1, sigma=5)
    # y2 = gaussian_filter1d(y2, sigma=5)
    # y3 = gaussian_filter1d(y3, sigma=5)
    # y5 = gaussian_filter1d(y5, sigma=5)
    # y4 = gaussian_filter1d(y4, sigma=5)
    # y5 = gaussian_filter1d(y5, sigma=5)


    # 实现函数

    func1 = interpolate.interp1d(x, y1, kind='cubic')
    func2 = interpolate.interp1d(x, y2, kind='cubic')
    func3 = interpolate.interp1d(x, y3, kind='cubic')
    func4 = interpolate.interp1d(x, y4, kind='cubic')
    func5 = interpolate.interp1d(x, y5, kind='cubic')
    func6 = interpolate.interp1d(x, y6, kind='cubic')


    # 利用xnew和func函数生成ynew,xnew数量等于ynew数量
    y1 = func1(x)
    y2 = func2(x)
    y3 = func3(x)
    y4 = func4(x)
    y5 = func5(x)
    y6 = func6(x)



    ax.plot(x,y1, label='wl',marker='o', markersize=2,)  # 作y1 = x 图，并标记此线名为linear
    ax.plot(x,y2, label='ik-wl', marker='x', markersize=3)  # 作y2 = x^2 图，并标记此线名为quadratic
    ax.plot(x,y3, label='iterative-ik-wl_r_1', marker='^', markersize=2)  # 作y3 = x^3 图，并标记此线名为cubic
    ax.plot(x, y4, label='iterative-ik-wl_r_2', marker='^', markersize=2)  # 作y3 = x^3 图，并标记此线名为cubic

    ax.plot(x, y5, label='iterative-ik-wl_r_5', marker='^', markersize=2)  # 作y3 = x^3 图，并标记此线名为cubic
    ax.plot(x, y6, label='iterative-gk-wl', marker='^', markersize=2)  # 作y3 = x^3 图，并标记此线名为cubic
    ax.set_xlabel('Iterations')  # 设置x轴名称 x label
    ax.set_ylabel('Similarity')  # 设置y轴名称 y label
    ax.set_title(name)  # 设置图名为Simple Plot
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示
    plt.xticks(np.arange(0, h+1,5), [i for i in range(0, h+1,5)])
    plt.show()  # 图形可视化


def new_balanced(g,embedding):
    new=np.zeros_like(embedding)
    for ind in range(embedding.shape[0]):

        tar = get_neigbors(g, ind)[1]+[ind]
        new[ind] = np.mean(embedding[tar],axis=0)
    return new


def sim(embedding,predict_labels):

    from utils import group_partition
    group = group_partition(predict_labels)
    # sim = embedding.dot((embedding.T))
    res = []
    for i in group:
        temp = embedding[i]
        part = temp.dot(temp.T)
        np.fill_diagonal(part,0)
        n = part.shape[0]
        s = np.sum(part)/(n*(n-1))
        res.append(s)
    full = embedding.dot(embedding.T)
    np.fill_diagonal(full, 0)
    n = full.shape[0]

    res2 = np.sum(full)/(n*(n-1))
    return np.mean(res), np.mean(res2), np.mean(res) / np.mean(res2)
if __name__ == "__main__":

    path1 = 'E:/Graph Clustering/dataset/real_world data/'
    # path1 = 'E:/Graph Clustering/dataset/artificial data/'

    dataset = "cora"
    adj_mat, node_features, true_labels = load_data(path1, dataset)
    num_of_class = np.unique(true_labels).shape[0]
    deg = np.sum(adj_mat, axis=0)
    res = []
    G = nx.from_numpy_matrix(adj_mat)
    for r in range(1):
        best = -1
        score_wl = []
        score_ik = []
        score_iik = []
        nmi_wl = []
        nmi_ik = []
        nmi_iik = []
        sim_wl =[]
        sim_ik=[]
        sim_iik=[]

        score_iik2 = []
        score_iik5 = []
        nmi_iik2 = []
        nmi_iik5= []

        sim_iik2=[]
        sim_iik5=[]

        sim_gk=[]
        score_gk=[]
        nmi_gk=[]

        adj = adj_mat.copy()
        new_adj =create_adj_avg(adj_mat)
        tar_index = [i for i in range(node_features.shape[0])]

        psi=64
        t=200
        embedding = copy.deepcopy(node_features)
        embeddinggk = copy.deepcopy(node_features)
        embeddingik = IK_fm_dot(node_features, psi, 200)
        embeddingiik = copy.deepcopy(embeddingik)
        embeddingiik2 = copy.deepcopy(embeddingik)
        embeddingiik5 = copy.deepcopy(embeddingik)

        psi_li=[64,64,64,32,16]
        for h in range(65):
            print("======================================={}==========================================".format(h))

            ## wl
            smoothing = mad_value(embedding, adj_mat, distance_metric='cosine', digt_num=4, target_idx=np.array(tar_index))
            score_wl.append(smoothing)
            embedding = WL_noconcate_one(embedding, new_adj)
            emb = preprocessing.normalize(embedding,"l2")
            acc, nmi, f1, para, predict_labels = cmd.sc_linear(emb, 1, num_of_class, true_labels)
            nmi_wl.append(nmi)
            print('@WL: {} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))
            a, b, c = sim(emb, true_labels)
            sim_wl.append([a,b,c])

            ## gkwl
            from sklearn.kernel_approximation import Nystroem

            feature_map_nystroem = Nystroem(gamma=0.01, random_state=1, n_components=300)
            embeddinggk = feature_map_nystroem.fit_transform(embeddinggk)
            smoothinggk = mad_value(embeddinggk, adj_mat, distance_metric='cosine', digt_num=4, target_idx=np.array(tar_index))
            score_gk.append(smoothinggk)
            embeddinggk = WL_noconcate_one(embeddinggk, new_adj)
            emb = preprocessing.normalize(embeddinggk, "l2")
            acc, nmi, f1, para, predict_labels = cmd.sc_linear(emb, 1, num_of_class, true_labels)
            nmi_gk.append(nmi)
            print('@IKWL: {} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))

            a, b, c = sim(emb, true_labels)
            sim_gk.append([a, b, c])


            ## ikwl
            smoothingik = mad_value(embeddingik, adj_mat, distance_metric='cosine', digt_num=4, target_idx=np.array(tar_index))
            score_ik.append(smoothingik)
            embeddingik = WL_noconcate_one(embeddingik, new_adj)
            emb = preprocessing.normalize(embeddingik, "l2")
            acc, nmi, f1, para, predict_labels = cmd.sc_linear(emb, 1, num_of_class, true_labels)
            nmi_ik.append(nmi)
            print('@IKWL: {} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))

            a, b, c = sim(emb, true_labels)
            sim_ik.append([a,b,c])
            ## iikwl

            if h>0:
                embeddingiik = IK_fm_dot(embeddingiik,psi=32,t=200)
                embeddingiik = preprocessing.normalize(embeddingiik, "l2",axis=0)
            # if h>0:
            #     sss = embeddingiik.dot(embeddingiik.T)
            #     sss = adj_mat*sss
            #     sss =create_adj_avg(sss)
            #     embeddingiik2 = WL_noconcate_one(embeddingiik,sss)
            #     embeddingiik2 = preprocessing.normalize(embeddingiik2, "l2", axis=0)
            #
            #     embeddingiik = np.concatenate([embeddingiik, embeddingiik2], axis=1)
            smoothingiik = mad_value(embeddingiik, adj_mat, distance_metric='cosine', digt_num=4, target_idx=np.array(tar_index))
            score_iik.append(smoothingiik)
            embeddingiik = WL_noconcate_one(embeddingiik, new_adj)
            emb = preprocessing.normalize(embeddingiik, "l2")
            acc, nmi, f1, para, predict_labels = cmd.sc_linear(emb, 1, num_of_class, true_labels)
            nmi_iik.append(nmi)
            print('@IIKWL: {} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))
            a, b, c = sim(emb, true_labels)
            sim_iik.append([a, b, c])
            ##iikwl_r_2
            if h>0 and h %2 ==0:
                embeddingiik2 = IK_fm_dot(embeddingiik2, psi=32, t=200)
                embeddingiik2 = preprocessing.normalize(embeddingiik2, "l2", axis=0)
            # if h > 0:
            #     sss = embeddingiik.dot(embeddingiik.T)
            #     sss = adj_mat * sss
            #     sss = create_adj_avg(sss)
            #     embeddingiik2 = WL_noconcate_one(embeddingiik, sss)
            #     embeddingiik2 = preprocessing.normalize(embeddingiik2, "l2", axis=0)
            #
            #     embeddingiik = np.concatenate([embeddingiik, embeddingiik2], axis=1)
            smoothingiik2 = mad_value(embeddingiik2, adj_mat, distance_metric='cosine', digt_num=4, target_idx=np.array(tar_index))
            score_iik2.append(smoothingiik2)
            embeddingiik2 = WL_noconcate_one(embeddingiik2, new_adj)
            emb = preprocessing.normalize(embeddingiik2, "l2")
            acc, nmi, f1, para, predict_labels = cmd.sc_linear(emb, 1, num_of_class, true_labels)
            nmi_iik2.append(nmi)
            print('@IIKWL: {} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))
            a, b, c = sim(emb, true_labels)
            sim_iik2.append([a, b, c])
            ##iikwl_r_5
            if h>0 and h % 5 == 0:
                embeddingiik5 = IK_fm_dot(embeddingiik5, psi=32, t=200)
                embeddingiik5 = preprocessing.normalize(embeddingiik5, "l2", axis=0)
            # if h > 0:
            #     sss = embeddingiik.dot(embeddingiik.T)
            #     sss = adj_mat * sss
            #     sss = create_adj_avg(sss)
            #     embeddingiik2 = WL_noconcate_one(embeddingiik, sss)
            #     embeddingiik2 = preprocessing.normalize(embeddingiik2, "l2", axis=0)
            #
            #     embeddingiik = np.concatenate([embeddingiik, embeddingiik2], axis=1)
            smoothingiik5 = mad_value(embeddingiik5, adj_mat, distance_metric='cosine', digt_num=4, target_idx=np.array(tar_index))
            score_iik5.append(smoothingiik5)
            embeddingiik5 = WL_noconcate_one(embeddingiik5, new_adj)
            emb = preprocessing.normalize(embeddingiik5, "l2")
            acc, nmi, f1, para, predict_labels = cmd.sc_linear(emb, 1, num_of_class, true_labels)
            nmi_iik5.append(nmi)
            print('@IIKWL: {} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))
            a, b, c = sim(emb, true_labels)
            sim_iik5.append([a, b, c])
            print(smoothing, smoothingik, smoothingiik2)
            if h>4:

                ssim_wl=np.array(sim_wl)
                ssim_ik=np.array(sim_ik)
                ssim_iik=np.array(sim_iik)
                ssim_iik2 = np.array(sim_iik2)

                ssim_iik5 = np.array(sim_iik5)
                ssim_gk = np.array(sim_gk)


                sim_plot(ssim_wl[:,0],ssim_ik[:,0],ssim_iik[:,0],ssim_iik2[:,0],ssim_iik5[:,0],ssim_gk[:,0],'inner similarity')
                sim_plot(ssim_wl[:, 1], ssim_ik[:, 1], ssim_iik[:, 1], ssim_iik2[:, 1], ssim_iik5[:, 1],ssim_gk[:,1], 'global similarity')
                sim_plot(ssim_wl[:, 2], ssim_ik[:, 2], ssim_iik[:, 2],ssim_iik2[:, 2],ssim_iik5[:, 2],ssim_gk[:,2], 'ratio of inner similarity/global similarity')
                plot_twin(score_wl[:h], score_ik[:h], score_iik[:h],score_gk[:h], nmi_wl[:h], nmi_ik[:h], nmi_iik[:h],nmi_gk[:h], 'Mean Average Distance', 'NMI')
                plt.show()

            #
            # nmi_wl = [0.350, 0.392, 0.425, 0.458, 0.479, 0.485, 0.480, 0.486, 0.497, 0.504, 0.498, 0.496, 0.482, 0.478, 0.472, 0.468, 0.454, 0.451, 0.448, 0.443, 0.449,
            #            0.445, 0.446, 0.368, 0.355, 0.330, 0.352, 0.320, 0.319, 0.323, 0.322, 0.319, 0.323, 0.313, 0.325, 0.320]
            # nmi_ik = [0.443, 0.503, 0.519, 0.531, 0.535, 0.533, 0.536, 0.542, 0.538, 0.532, 0.533, 0.524, 0.523, 0.515, 0.514, 0.513, 0.511, 0.508, 0.507, 0.507, 0.503,
            #            0.502, 0.501, 0.496, 0.495, 0.488, 0.478, 0.477, 0.476, 0.475, 0.475, 0.473, 0.467, 0.465, 0.463, 0.448]
            # nmi_iik = [0.451, 0.545, 0.562, 0.571, 0.565, 0.561, 0.558, 0.554, 0.548, 0.551, 0.548, 0.547, 0.543, 0.545, 0.542, 0.541, 0.550, 0.543, 0.545, 0.547, 0.548,
            #               0.545, 0.541, 0.535, 0.541, 0.536, 0.531, 0.539, 0.530, 0.525, 0.521, 0.515, 0.518, 0.513, 0.508, 0.511, ]
            # print(smoothing,smoothingik,smoothingiik)
            #




            ## 计算IDK分组
            # drop_li = []
            # group = []
            # for i in range(num_of_class):
            #     temp = np.where(predict_labels == i)[0].tolist()
            #     group.append(temp)
            # idk = np.zeros(node_features.shape[0])
            # sort_id = []
            # for i in range(len(group)):
            #     member_id = group[i]
            #     member_emb = embedding[member_id]
            #     member_idk = IDK(member_emb,8,t)
            #     for t in range(len(member_id)):
            #         id_t = member_id[t]
            #         idk[id_t] = member_idk[t]
            #
            #     ind = np.argsort(member_idk).tolist()
            #
            #     ind = [group[i][id] for id in ind]
            #     sort_id.extend(ind)
            # sort_id = np.array(sort_id)
            # sort_idk = []
            # truth = []
            # cnt = 0
            #
            # for i in sort_id:
            #     sort_idk.append(idk[i])
            #     if predict_labels[i] == true_labels[i]:
            #         truth.append(0)
            #         cnt += 1
            #     else:
            #         truth.append(-(true_labels[i] + 1) * 0.1)
            # sort_idk = np.array(sort_idk)
            #
            # idk_li.append(sort_idk)
            # pre_li = truth
            # x = [i for i in range(node_features.shape[0])]
            # plt.bar(range(len(truth)), truth)
            # plt.plot(x, sort_idk, color="red", linewidth=1, linestyle='dashdot')
            # plt.title("@h={}".format(h))
            # plt.show()
            #
            #
            # group_flag = [0]
            # flag = 0
            # for i in range(num_of_class):
            #     flag += len(group[i])
            #     group_flag.append(flag)
            # idk_flag=[]
            # # for i in range(num_of_class):
            # #     flag1 = group_flag[i]
            # #     flag2 = group_flag[i+1]
            # #     ttttt = 0
            # #     for j in range(flag1+1,flag2-1):
            # #         idj1 = sort_id[j]
            # #         idj2 = sort_id[j-1]
            # #         idj3 = sort_id[j+1]
            # #         if true_labels[idj1]!= predict_labels[idj1] and true_labels[idj2]!= predict_labels[idj2] and true_labels[idj3]!= predict_labels[idj3] and sort_idk[idj3]<np.mean(sort_idk):
            # #             ttttt = idj3
            # #     idk_flag.append(sort_idk[ttttt])
            #
            # nm = np.mean(sort_idk)
            # nmm = np.min(sort_idk)
            # mmm = np.max(sort_idk)
            # idk_flag.append(nm)
            #
            # # idk_flag = []
            # th_li=[0.55,0.55,0.6,0.6,0.55,0.55,0.6,0.6,0.55]
            # th = th_li[h-1]
            #
            # if h>0:
            #     adj_mat = adj.copy()
            #     drop_li2 = np.where(sort_idk > th)[0]
            #     drop_li2 = [sort_id[i] for i in drop_li2]
            #     # drop_li = [i for i in range(node_features.shape[0]) if i not in drop_li]
            #
            #
            #     adj_mat[drop_li2] = 0
            #     adj_mat[:, drop_li2] = 0
            #     am2 = len(drop_li2)
            #
            #     # drop_li1 = np.where(sort_idk < th)[0]
            #     # drop_li1 = [sort_id[i] for i in drop_li1]
            #
            #     #
            #     # # adj_mat = adj.copy()
            #     # adj_mat[drop_li1] = 1
            #     # adj_mat[:,drop_li1] = 1
            #     # am1 =len(drop_li1)
            #
            #     # embedding1 = WL_noconcate_one(embedding, adj_mat)
            #     # embedding1 = preprocessing.normalize(embedding1, 'l2')
            #     # embedding1 = WL_noconcate_one(embedding1, adj_mat)
            #     # embedding1 = preprocessing.normalize(embedding1, 'l2')
            #     # embedding1 = WL_noconcate_one(embedding1, adj_mat)
            #     # embedding1 = preprocessing.normalize(embedding1, 'l2')
            #     # embedding1 = WL_noconcate_one(embedding1, adj_mat)
            #     # embedding1 = preprocessing.normalize(embedding1, 'l2')
            #
            #     # embedding1 = IK_fm_dot(embedding, 64, 200)
            #
            #     embedding1 = WL_noconcate_one(embedding, adj_mat)
            #     embedding1 = preprocessing.normalize(embedding1, 'l2')
            #     # embedding2 = WL_noconcate_one(embedding2, adj_mat)
            #     # embedding2 = preprocessing.normalize(embedding2, 'l2')
            #     # embedding2 = WL_noconcate_one(embedding2, adj_mat)
            #     # embedding2 = preprocessing.normalize(embedding2, 'l2')
            #
            #     # adj_mat = adj.copy()
            #     # drop_li2 = [i for i in range(node_features.shape[0]) if i not in drop_li+drop_li1]
            #     #
            #     # adj_mat[drop_li2] = 0
            #     # adj_mat[:, drop_li2] = 0
            #     # embedding3 = WL_noconcate(embedding, adj_mat, s)
            #     # embedding3 = preprocessing.normalize(embedding3, 'l2')
            #
            #
            #     # print("egde1:{} egde2:{} idk_mean:{} idk_in:{} rate:{} smoothing:{}".format(am2,am1,nm,nmm,nm,smoothing ))
            #
            #     adj_mat =adj.copy()
            #     # embedding2 = WL_noconcate(embedding, adj_mat, 3)
            #     # embedding2 = preprocessing.normalize(embedding2, 'l2')
            #
            #     embedding = 0.9*embedding1+0.1*embedding
            #
            #     # embedding = preprocessing.normalize(embedding, 'l2')
            #
            #     aacc, nnmi, ff1, para, predict_labels = cmd.sc_linear(embedding, 1, num_of_class, true_labels)
            #
            #     print('+++++++++++: ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(aacc, nnmi, ff1))
            #     #######
    #
    #     print("best={}".format(best))
    #     colors = ['red', 'blue', 'green', 'hotpink', 'purple', 'black', 'yellow', "grey", 'orange', "indigo", "teal", "navy", "skyblue", "aqua", "gold"]
    #     x = [i for i in range(node_features.shape[0])]
    #
    #     # plt.scatter(x, pre_li, s=1, c=pre_li,marker='o', alpha=0.5)
    #     for i in range(0, len(idk_li), 2):
    #         idks = idk_li[i]
    #         color = colors[i]
    #         plt.plot(x, idks, color=color, linewidth=1, linestyle='dashdot')
    #
    #     plt.show()
    #     # acc, nmi, f1, para, predict_labels = cmd.sc_semi(adj_mat,embedding, 1, num_of_class, true_labels)
    #     # pplot2(node_features, embedding, f'h={h}', true_labels, predict_labels, p=100)
    #     #     print('@{} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))
    #     #     if best<nmi:
    #     #         best=nmi
    #
    # print("@ave: best={}".format(np.mean(res)))
    #
