import networkx as nx
import matplotlib.pyplot as plt
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
import heapq
import numpy as np
import sklearn.cluster as sc
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
    feature_mean_map = np.mean(point_fm_list,axis=0)
    idk_score = np.dot(point_fm_list, feature_mean_map) / t
    # idk_score = (idk_score - np.min(idk_score)) / (np.max(idk_score) - np.min(idk_score))
    # idk_score = (idk_score - np.mean(idk_score)) / np.std(idk_score)
    idk_score = (idk_score - np.min(idk_score)) / (np.max(idk_score) - np.min(idk_score))
    # idk_score = idk_score / np.max(idk_score)

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
    single_neighbors = get_neighbors_single(G, node, 3)
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


def plot_graph(G,emb,label,sample_list):

    tsne = TSNE(n_components=2, perplexity=50, n_iter=1000)
    emb_tsne = tsne.fit_transform(emb)
    mapping = {}

    for i, j in enumerate(sample_list):
        mapping[j] = i
    G = nx.relabel_nodes(G, mapping)

    nx.draw(G,emb_tsne,node_color=label,node_size=15)
    plt.show()





def mad_value(in_arr, mask_arr, distance_metric='cosine', digt_num=4, target_idx=None):
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
def plot_twin(_y1, _y2,_y3, _y4, _ylabel1, _ylabel2):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('iterations (h)')
    ax1.set_ylabel(_ylabel1, color=color)
    ax1.plot(_y1, color=color, marker='s',label='WL-mad')
    ax1.plot(_y2, color=color, marker='^',label='IKWL-mad' )
    ax1.tick_params(axis='y', labelcolor=color)
    plt.legend()  # 让图例生效

    ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴

    color = 'tab:orange'
    ax2.set_ylabel(_ylabel2, color=color)
    ax2.plot(_y3, color=color, marker='s',label='WL-NMI')
    ax2.plot(_y4, color=color, marker='^',label='IKWL-NMI')

    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.legend()  # 让图例生效

    # plt.show()

def group_partition(predict_labels):
    ## 获取每个group的下标
    num_of_class = np.unique(predict_labels).shape[0]
    group_of_pre = []
    for i in range(num_of_class):
        temp = np.where(predict_labels == i)[0].tolist()
        group_of_pre.append(temp)
    return group_of_pre
def IKWL(node_features,adj_mat,psi,h=3):
    new_adj =adj_mat+adj_mat.dot(adj_mat)
    new_adj = np.where(new_adj>0,1,0)
    embedding_wl = WL_noconcate_one(node_features, new_adj)
    embedding_ikwl1 = IK_fm_dot(embedding_wl,7,t)
    # embedding_ikwl2 = IK_fm_dot(embedding_wl, 7, t)
    # embedding_ikwl = np.concatenate((embedding_ikwl1,embedding_ikwl2),axis=1)
    return embedding_ikwl1

def id_sort_by_idk(group_of_pre,):


    sort_id = []
    idk_score = np.zeros(node_features.shape[0])

    for i in range(len(group_of_pre)):
        member_id = group_of_pre[i]
        member_emb = embedding[member_id]
        member_idk = IDK(member_emb, 2, 100)
        for t in range(len(member_id)):
            id_t = member_id[t]
            idk_score[id_t] = member_idk[t]

        id_in_group_sored = np.argsort(member_idk).tolist()

        ind = [group_of_pre[i][id] for id in id_in_group_sored]
        sort_id.extend(ind)
    return idk_score,np.array(sort_id)
if __name__ == "__main__":

    path1 = 'E:/Graph Clustering/dataset/real_world data/'
    dataset = "cora"
    adj_mat, node_features, true_labels = load_data(path1, dataset)
    num_of_class = np.unique(true_labels).shape[0]
    deg = np.sum(adj_mat, axis=0)
    res = []
    G = nx.from_numpy_matrix(adj_mat)
    for r in range(1):
        psi = 64
        t = 100
        print("=========================================={}=============================================".format(r))
        best = -1
        idk_li = []
        pre_li = []
        score1 = []
        score2 = []
        nmi_li1 = []
        nmi_li2 = []
        embedding = node_features.copy()
        embedding = IK_fm_dot(embedding,psi,t)
        adj = adj_mat.copy()

        psili =[64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]
        for h in range(65):
            psi = psili[h]

##################################################### IKWL #############################################################
            embedding0 = copy.deepcopy(embedding)
            embedding = IKWL(embedding,adj_mat,psi,h=1)

            embedding = preprocessing.normalize(embedding,"l2")
            acc, nmi, f1, para, predict_labels = cmd.sc_linear(embedding, 1, num_of_class, true_labels)


            tsne = TSNE(n_components=2, perplexity=65)
            node_features_tsne = tsne.fit_transform(node_features)

            tsne = TSNE(n_components=2, perplexity=65)
            embedding_tsne = tsne.fit_transform(embedding)
            pplot2(node_features_tsne, embedding_tsne, f'h={h}', true_labels, predict_labels, p=1000)

            print('@{} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))
            tar_index = [i for i in range(node_features.shape[0])]
            smoothing = mad_value(embedding, adj_mat, distance_metric='cosine', digt_num=4, target_idx=np.array(tar_index ))
            score1.append(smoothing)
            nmi_li1.append(nmi)
            # plot_twin(score1,score2, nmi_li1,nmi_li2, 'value of non-smoothing', 'nmi')
            # plt.show()
#########################################################################################################################

##################################################### 按预测分组 ###########################################################
            group_of_pre = group_partition(predict_labels)
            idk_score,sort_id=id_sort_by_idk(group_of_pre)

            sort_idk = []
            truth = []
            cnt = 0
            drop_li = []
            for i in sort_id:
                sort_idk.append(idk_score[i])
                if predict_labels[i] == true_labels[i]:
                    truth.append(0)
                    cnt += 1
                else:
                    truth.append(-(true_labels[i] + 1) * 0.1)
            sort_idk = np.array(sort_idk)

            idk_li.append(sort_idk)
            pre_li = truth
            x = [i for i in range(node_features.shape[0])]
            plt.bar(range(len(truth)), truth)
            plt.plot(x, sort_idk, color="red", linewidth=1, linestyle='dashdot')
            plt.title("@h={}".format(h))
            plt.show()


            group_flag = [0]
            flag = 0
            for i in range(num_of_class):
                flag += len(group_of_pre[i])
                group_flag.append(flag)


            mean_idk = np.mean(sort_idk)
            min_idk = np.min(sort_idk)
            max_idk = np.max(sort_idk)
##################################################### wl chai ###########################################################
            ##################



            ##################

            ##################################################### wl chai1 ###########################################################
            print(np.mean(sort_idk),np.median(sort_idk))
            th = 0.5
            adj_mat = copy.deepcopy(adj)
            drop_li1 = np.where(sort_idk < th)[0]
            drop_li1 = [sort_id[i] for i in drop_li1]
            th = 0.5
            drop_li2 = np.where(sort_idk > th)[0]
            drop_li2 = [sort_id[i] for i in drop_li2]
            sim = embedding.dot(embedding.T)
            # drop_li = [i for i in range(node_features.shape[0]) if i not in drop_li]
            for i in drop_li1:
                for j in drop_li2:



                    adj_mat[j][i] = 0
            #
            # for i in drop_li1:
            #     jid = getListMaxNumIndex(sim[i],1)
            #     adj_mat[i,:] = 0
            #     adj_mat[i][jid] = 1




            print(np.sum(adj_mat)/2)
            # embedding2 = IK_fm_dot(embedding,64,t)
            embedding1 = WL_noconcate_one(embedding0,adj_mat)
            embedding_final = preprocessing.normalize(embedding1,'l2')
            embedding2 = WL_noconcate_one(embedding, adj_mat)
            embedding_final2 = preprocessing.normalize(embedding2, 'l2')

            aacc, nnmi, ff1, para, predict_labels = cmd.sc_linear(embedding1, 1, num_of_class, true_labels)
            print('+++++++++++: ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(aacc, nnmi, ff1))

            aacc, nnmi, ff1, para, predict_labels = cmd.sc_linear(embedding2, 1, num_of_class, true_labels)

            tsne = TSNE(n_components=2, perplexity=65)
            node_features_tsne = tsne.fit_transform(node_features)

            tsne = TSNE(n_components=2, perplexity=65)
            embedding_tsne = tsne.fit_transform(embedding_final)
            pplot2(node_features_tsne, embedding_tsne, f'h={h}', true_labels, predict_labels, p=1000)
            print('+++++++++++: ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(aacc, nnmi, ff1))

            embedding_final = np.concatenate((embedding,embedding_final,embedding_final2),axis=1)

            embedding =embedding_final
            adj_mat = copy.deepcopy(adj)
##################################################### wl chai2 ###########################################################





            #######

        print("best={}".format(best))
        colors = ['red', 'blue', 'green', 'hotpink', 'purple', 'black', 'yellow', "grey", 'orange', "indigo", "teal", "navy", "skyblue", "aqua", "gold"]
        x = [i for i in range(node_features.shape[0])]

        # plt.scatter(x, pre_li, s=1, c=pre_li,marker='o', alpha=0.5)
        for i in range(0, len(idk_li), 2):
            idks = idk_li[i]
            color = colors[i]
            plt.plot(x, idks, color=color, linewidth=1, linestyle='dashdot')

        plt.show()
        # acc, nmi, f1, para, predict_labels = cmd.sc_semi(adj_mat,embedding, 1, num_of_class, true_labels)
        # pplot2(node_features, embedding, f'h={h}', true_labels, predict_labels, p=100)
        #     print('@{} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))
        #     if best<nmi:
        #         best=nmi
        # res.append(best)

        print("@rep={} best:{}".format(r, nmi))
    print("@ave: best={}".format(np.mean(res)))

