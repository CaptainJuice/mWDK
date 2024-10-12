import copy
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score,normalized_mutual_info_score,accuracy_score
from munkres import Munkres
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
import clustering_methods as cmd
from utils import GSNN, load_data,WL, WL_noconcate_one,WL_noconcate, IGK_WL_noconcate,IK_inne_fm,IK_fm_dot,WL_noconcate_gcn,pplot2,pplot3,pplot_single,pplot_single_seperate
from utils import create_adj_avg,WL_noconcate_fast,create_adj_avg_gcn,create_adj_avg_sp,adj_plot
from sklearn.kernel_approximation import Nystroem
import warnings
import scipy.io as sio
from sklearn import preprocessing
from Lambda_feature import lambda_feature_continous
from sklearn.metrics.pairwise import pairwise_distances

warnings.filterwarnings('ignore')

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

def smooth(embedding,labels):
    embs=IK_fm_dot(embedding,32,100)
    resi,resj,dis=-1,-1,np.inf
    # sim = pairwise_distances(embedding, metric="cosine")
    sim=embs.dot(embs.T)
    class0= [i for i in range(len(labels)) if labels[i]==0]
    class1= [i for i in range(len(labels)) if labels[i]==1]
    class2=[i for i in range(len(labels)) if labels[i]==2]
    class3= [i for i in range(len(labels)) if labels[i]==3]
    class4=[i for i in range(len(labels)) if labels[i]==4]
    class5= [i for i in range(len(labels)) if labels[i]==5]
    class6=[i for i in range(len(labels)) if labels[i]==6]
    # for i in class1:
    #     for j in class2:
    #         if sim[i][j]<dis:
    #             dis =sim[i][j]
    #             resi=i
    #             resj =j

    s = [embs[j] for j in class0]
    s0 = np.mean(s, axis=0)
    s=[embs[i] for i in class1]
    s1=np.mean(s,axis=0)
    s=[embs[j] for j in class2]
    s2=np.mean(s,axis=0)
    s = [embs[j] for j in class3]
    s3 = np.mean(s, axis=0)
    s = [embs[j] for j in class4]
    s4 = np.mean(s, axis=0)
    s = [embs[j] for j in class5]
    s5 = np.mean(s, axis=0)
    s = [embs[j] for j in class6]
    s6 = np.mean(s, axis=0)


    a0,a1,a2,a3,a4,a5,a6=0,0,0,0,0,0,0
    for i in class1:
        a0 += embs[i].dot(s0.T)
    for i in class1:
        a1 += embs[i].dot(s1.T)
    for i in class1:
        a2 += embs[i].dot(s2.T)
    for i in class1:
        a3 += embs[i].dot(s3.T)
    for i in class1:
        a4 += embs[i].dot(s4.T)
    for i in class1:
        a5 += embs[i].dot(s5.T)
    for i in class1:
        a6 += embs[i].dot(s6.T)

    ss=[s0,s1,s2,s3,s4,s5,s6]
    ss=np.array(ss)
    ss_all=np.mean(embs,axis=0)
    # ss=preprocessing.normalize(ss,"l2")
    ss=ss.dot(ss_all.T)







    # dis2=np.mean(sim)
    # return dis,resi,resj,dis2,dis/dis2
    return np.mean(ss),a0/len(class0),a1/len(class1),a2/len(class2),a3/len(class3),a4/len(class4),a5/len(class5),a6/len(class6)
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

def weight(G,embedding):
    # mean_emb = np.zeros_like(embedding,dtype=float)
    # neighbors_list = [get_neigbors(G, tar, depth=1)[1] for tar in range(node_features.shape[0])]
    # for i in range(node_features.shape[0]):
    #     neighbors = neighbors_list[i]
    #     mean_emb[i] = np.mean(embedding[neighbors],axis=0)
    # mean_emb=preprocessing.normalize(mean_emb,"l2")
    # sim =mean_emb.dot(mean_emb.T)
    # np.fill_diagonal(sim,0)
    # sim = preprocessing.normalize(sim, "l2",axis=1)
    #
    # sim = 0.5*np.log2((1-sim)/sim)
    # # sim =1/sim
    # sim = np.where(sim==np.inf,0,sim)

##################################################### version2
    # mean_emb = np.mean(embedding,axis=0)
    # sim = embedding.dot(mean_emb)
    # sim =(1+np.exp(sim))
    # #
    #
    # sim=np.diag(sim)
#################################################### version3
    mean_emb = np.zeros_like(embedding, dtype=float)
    neighbors_list = [get_neigbors(G, tar, depth=1)[1] for tar in range(node_features.shape[0])]
    for i in range(node_features.shape[0]):
        neighbors = neighbors_list[i]
        mean_emb[i] = np.mean(embedding[neighbors], axis=0)

    mean_emb = IK_fm_dot(mean_emb,32,100)
    sim = mean_emb.dot(mean_emb.T)
    # sim =np.diag(np.diag(sim))

    return sim,mean_emb


def normalize_adj(adj, type='rw'):
    np.fill_diagonal(adj, 1)
    """Symmetrically normalize adjacency matrix."""

    if type == 'sym':

        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        return adj*d_inv_sqrt*d_inv_sqrt.flatten()
        # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # d_mat_inv_sqrt = sp.diags(d_inv_sqrt).todense()
        # return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized
    elif type == 'wl':


        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = np.diag(d_inv)
        adj_normalized = adj.dot(d_mat_inv)
        return adj_normalized

def create_adj_avg_temp(adj_mat,sim):
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
    # np.fill_diagonal(sim,0)
    # sim = preprocessing.normalize(sim,'l2')
    adj = adj *sim
    np.fill_diagonal(adj,0.5)
    return adj

def group_partition(predict_labels):
## 获取每个group的下标
    num_of_class = np.unique(predict_labels).shape[0]
    group_of_pre = []
    for i in range(num_of_class):
        temp = np.where(predict_labels == i)[0].tolist()
        group_of_pre.append(temp)
    return group_of_pre

def WL_max(embedding,adj,gg):
    new_emb = copy.deepcopy(embedding)
    embedding=embedding.astype(float)
    for i in range(adj.shape[0]):
        nei = get_neigbors(gg,i)[1]+[i]

        temp = embedding[nei]
        res = np.max(temp,axis=0)
        new_emb[i] = res
    return new_emb.astype(float)
def WL_min(embedding,adj,gg):
    new_emb = copy.deepcopy(embedding)
    embedding=embedding.astype(float)
    for i in range(adj.shape[0]):
        nei = get_neigbors(gg,i)[1]+[i]

        temp = embedding[nei]
        res = np.min(temp,axis=0)
        new_emb[i] = res
    return new_emb.astype(float)
if __name__ == '__main__':
    ###################################### parm ########################################
    emb_type = {"wl_noconcate":1,
              "ikwl_noconcate":12,
              "new_ikwl_noconcate":13,
            "new_gkwl_noconcate": 12
               }
    path1 = 'E:/Graph Clustering/dataset/real_world data/'
    datasets1 = ['cora','citeseer','wiki','dblp',]
    datasets2 = ['acm','blogcatalog','flickr','dblp','amac','amap']
    # path1 = 'E:/Graph Clustering/dataset/artificial data/'
    datasets = ['cora','blogcatalog']
    # datasets= datasets1

    for dataset in datasets:
        adj_mat, node_features, true_labels = load_data(path1, dataset)
        deg = np.sum(adj_mat,axis=0)
        print(dataset,np.max(deg),np.mean(deg),np.min(deg))

    # datasets = ['amac','amap']
    ####################################################################################

    for dataset in datasets:
        adj_mat, node_features, true_labels = load_data(path1, dataset)
        # adj_plot(adj_mat, true_labels, 100)
        G = nx.from_numpy_matrix(adj_mat)
        res = []
        #######################
        # mydic = {}
        # # group = group_partition(true_labels)
        # for i in range(len(true_labels)):
        #     mydic[i] = true_labels
        # \
        # cnt, cmt = 0, 0
        # for i in range(adj_mat.shape[0]):
        #     for j in range(adj_mat.shape[1]):
        #         if true_labels[i] != true_labels[j]:
        #             cnt+=1
        #             if adj_mat[i][j]!=0:
        #                 cmt+=1
        # print(dataset,cnt,cmt,cmt/cnt)






        ################
    #     for i in range(adj_mat.shape[0]):
    #         nei = get_neigbors(G,i,depth=1)[1]
    #         lab = [true_labels[i]]
    #         lab.extend(true_labels[nei])
    #         temp = (len(np.unique(lab))-1)/len(lab)
    #         res.append(temp)
    #     print('dataset:{}'.format(dataset),np.mean(res))
    #     from utils import group_partition
    #     group = group_partition(true_labels)
    #     deg = np.sum(adj_mat,axis=1)
    #     dli = []
    #     fli = []
    #     gli = []
    #     t=100
    #     IK_fea = IK_fm_dot(node_features,32,t)
    #     from sklearn.kernel_approximation import Nystroem
    #
    #     num_of_class = np.unique(true_labels).shape[0]
    #
    #     feature_map_nystroem = Nystroem(gamma=0.07, )
    #     GK_fea = feature_map_nystroem.fit_transform(node_features)
    #     acc, nmi, f1, para, predict_labels = cmd.km(GK_fea, 1, num_of_class, true_labels)
    #     print('pppp:',acc,nmi)
    #     for g in group:
    #         temp_deg = deg[g]
    #         avg_deg = np.mean(temp_deg)
    #         dli.append(avg_deg)
    #
    #         temp_fea = IK_fea[g]
    #         mean_fea = np.mean(temp_fea,axis=0)
    #         density = temp_fea.dot(mean_fea.T)/t
    #         avg_density = np.mean(density)
    #         fli.append(avg_density)
    #
    #         temp_fea = GK_fea[g] # 取出一个cluster的point
    #         mean_fea = np.mean(temp_fea,axis=0) # 求 feature mean map
    #         density = temp_fea.dot(mean_fea.T) # 点积
    #         avg_density = np.mean(density)
    #         gli.append(avg_density)
    #
    #
    #     print(max(dli),min(dli))
    #     print(max(fli), min(fli),(max(fli)/min(fli)))
    #     print(max(gli), min(gli),(max(gli)/min(gli)))
    #     # print(1/0)
    # from pylab import *

    # 这里可以调整图片大小
    # plt.figure(figsize=(18, 9))
    # x轴
    # x_axis_data = [i for i in range(1, 11)]  # 第一条折
    # B09B1LLLBG_data = [1.68, 5.07, 8.53, 12.11, 15.81, 20.86, 24.54, 27.17, 30.38, 35.62]
    #
    # B0091OMWUW_data = [1.63, 5.88, 13.54, 25.55, 39.74, 56.54, 76.47, 101.31, 128.01, 158.43, ]
    #
    # # # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    # # plt.plot(x_axis_data, B09B1LLLBG_data, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='the rank of B09B1LLLBG ')
    #
    # plt.plot(x_axis_data, B0091OMWUW_data, marker='o', mec='r', mfc='w', label='mWDK(concat)')
    # plt.plot(x_axis_data, B09B1LLLBG_data, marker='*', ms=10, label='mWDK(non-concat) ')
    # plt.legend()  # 让图例生效
    #
    # # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    # plt.legend(loc="upper left")
    # plt.xlabel('Iterations h')
    # plt.ylabel('Runtime(s)')


    rep =1
    if emb_type['wl_noconcate'] == 1:#312
        for dataset in datasets:

            adj_mat, node_features, true_labels = load_data(path1, dataset)
            # node_features= np.load('NDLS-main/src/cora_ndls.npy')

            # pa = 'E:\\Graph Clustering\\GraphSNN-main\\Node_Classification\\3.GraphSN_GAT\\{}_gsnn.npy'.format(dataset)
            # adj_mat = np.load(pa)
            # adj_plot(adj_mat,true_labels,100)

            adj2=copy.deepcopy(adj_mat)
            num_of_class = np.unique(true_labels).shape[0]
            # np.where(adj_mat != 0, adj_mat, 1)

            acc_li,nmi_li,f1_li = [],[],[]
            for r in range(rep):
                best_acc, best_nmi, best_f1, best_h= -1, -1, -1, -1
                emb = node_features.copy()
                time_start = time.perf_counter()
                embedding = node_features.copy()
                new_adj = create_adj_avg(adj_mat)
                res = []
                # embedding = IK_fm_dot(embedding, psi=64, t=200)
                embedding1 = WL_noconcate_fast(embedding, new_adj)


                def generate_hnodes(h_adj):
                    h_adj = h_adj.tocoo()
                    h_index = [[] for i in range(h_adj.shape[0])]
                    for i, j in zip(h_adj.row, h_adj.col):
                        h_index[i].append(j)
                    return h_index


                def generate_hadj(adj, h):
                    adj_h = sp.eye(adj.shape[0])
                    adj_tot = sp.eye(adj.shape[0])
                    for i in range(h):
                        adj_h = adj_h * adj
                        adj_tot = adj_tot + adj_h
                    return adj_tot


                def createWlEmbedding(node_features, adj_mat, h):
                    graph_feat = []
                    for it in range(h + 1):
                        if it == 0:
                            graph_feat.append(node_features)
                        else:
                            adj_cur = adj_mat + np.identity(adj_mat.shape[0])

                            adj_cur = create_adj_avg_ano(adj_cur)

                            np.fill_diagonal(adj_cur, 0)
                            graph_feat_cur = 0.5 * \
                                             (np.dot(adj_cur, graph_feat[it - 1]) + graph_feat[it - 1])
                            graph_feat.append(graph_feat_cur)
                    return np.mean(np.concatenate(graph_feat, axis=1), axis=0)
                    # return np.mean(graph_feat_cur, axis=0)


                # Generate h_nodes and their height
                def generate_h_nodes_n_dict(adj, h):
                    adj_h = sp.eye(adj.shape[0])
                    M = [{i: 0} for i in range(adj.shape[0])]
                    h_index = [[i] for i in range(adj.shape[0])]
                    for _ in range(h):
                        adj_h = sp.coo_matrix(adj_h * adj)

                        for i, j in zip(adj_h.row, adj_h.col):
                            if j in M[i]:
                                continue
                            else:
                                M[i][j] = _ + 1
                                h_index[i].append(j)
                    return M, h_index


                def generate_subgraph_embeddings(attr, adj, subgraph_index, h):
                    embedding = []
                    for i in range(adj.shape[0]):
                        root_feature = attr[i, :]
                        feature = attr[subgraph_index[i]]
                        feature = feature - np.tile(root_feature, (len(subgraph_index[i]), 1))
                        adj_i = adj[subgraph_index[i], :][:, subgraph_index[i]]
                        embedding.append(createWlEmbedding(feature, adj_i, h).reshape(1, -1))
                    return np.concatenate(embedding, axis=0)


                def create_adj_avg_ano(adj_cur):
                    '''
                    create adjacency
                    '''
                    deg = np.sum(adj_cur, axis=1)
                    deg = np.asarray(deg).reshape(-1)

                    deg[deg != 1] -= 1

                    deg = 1 / deg
                    deg_mat = np.diag(deg)
                    adj_cur = adj_cur.dot(deg_mat.T).T

                    return adj_cur
                def subgraph_embeddings(attr, adj, h):
                    M, h_index = generate_h_nodes_n_dict(adj, h)
                    embedding = generate_subgraph_embeddings(attr, adj, h_index, h)
                    return embedding, M


                from scipy.sparse import csc_matrix

                embedding1 = IK_inne_fm(embedding,psi=2,t=100)
                # embedding1 = WL_noconcate_fast(embedding1, new_adj)
                # embedding1 = WL_noconcate_fast(embedding1, new_adj)
                # embedding1 = WL_noconcate_fast(embedding1, new_adj)
                # embedding1 = WL_noconcate_fast(embedding1, new_adj)
                # embedding1 = WL_noconcate_fast(embedding1, new_adj)
                # embedding1 = WL_noconcate_fast(embedding1, new_adj)
                # embedding1 = WL_noconcate_fast(embedding1, new_adj)
                adj_csc = csc_matrix(adj_mat)
                ses,M= subgraph_embeddings(embedding1,adj_csc,h=3)
                ses = np.array(ses)
                kmembeddings = IK_inne_fm(ses,psi=4,t=300)

                mean_embedding = np.mean(kmembeddings, axis=0)
                scores = kmembeddings.dot(mean_embedding.transpose())
                # embedding1 = WL_noconcate_fast(embedding1, new_adj)
                # embedding1 = WL_noconcate_fast(embedding1, new_adj)
                # embedding1 = WL_noconcate_fast(embedding1, new_adj)
                # embedding1 = WL_noconcate_fast(embedding1, new_adj)
                # # embedding = WL_noconcate_fast(embedding, new_adj)
                # for i in range(node_features.shape[0]):
                #
                #         nei = get_neigbors(G, i, depth=1)[1]
                #         emb = embedding1[nei]
                #         new_emb = emb - embedding1[i]
                #         res_emb = np.mean(new_emb,axis=0)
                #         res.append(res_emb)
                #         # print()
                # embedding_ano = np.array(res)
                #
                # kmembeddings = IK_inne_fm(embedding_ano,2,100)
                # mean_embedding = np.mean(kmembeddings, axis=0)
                # scores = kmembeddings.dot(mean_embedding.transpose())


                def find_max_m_indices(nums, m):
                    nums = list(nums)
                    # 使用sorted函数和lambda表达式对元素进行排序，同时保留原始下标
                    sorted_with_indices = sorted(enumerate(nums), key=lambda x: x[1], reverse=True)
                    # 使用列表推导式提取前m个元素的下标
                    indices = [idx for idx, _ in sorted_with_indices[:m]]
                    return indices
                ano = find_max_m_indices(scores,100)
                from sklearn.ensemble import IsolationForest
                # clf = IsolationForest(contamination=0.05)
                #
                # # 使用训练数据拟合模型
                # clf.fit(embedding_ano)
                #
                # # 预测新的数据点
                #
                # y_pred = clf.predict(embedding_ano)
                # ano = list(np.where(y_pred==-1))

                # 输出预测结果，-1表示异常点，1表示正常点
                print(len(ano))
                matrix_without_rows = np.delete(adj_mat, ano, axis=0)

                # 从上一步的结果中删除指定的列
                new_adj_del = np.delete(matrix_without_rows, ano, axis=1)
                embedding_del = np.delete(embedding, ano, axis=0)
                new_adj_del = create_adj_avg(new_adj_del )
                true_labels_del = np.delete(true_labels, ano, axis=0)
                print(len(true_labels_del))
                for h in range(100):
                    embedding_del = IK_fm_dot(embedding_del, psi=64, t=200)
                    embedding_del = WL_noconcate_fast(embedding_del, new_adj_del)
                    embedding_del = preprocessing.normalize(embedding_del, norm='l2', axis=0, )
                    acc, nmi, f1, para, predict_labels = cmd.sc_linear(embedding_del, 1, num_of_class, true_labels_del)
                    print('@{} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))
                    embedding = IK_fm_dot(embedding, psi=64, t=200)
                    for o in range(adj_mat.shape[0]):
                        if o in ano:
                            del_o = ano.index(o)
                            embedding[o] = embedding_del[del_o]



                    embedding = WL_noconcate_fast(embedding, new_adj)
                    embedding= preprocessing.normalize(embedding, norm='l2', axis=0, )
                    acc, nmi, f1, para, predict_labels = cmd.sc_linear(embedding, 1, num_of_class, true_labels)
                    print('@new{} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, h, para, acc, nmi, f1))

                        # for h in range(100):
                #
                #     if h >0:
                #         get_neigbors(G,i,depth=1)
                #         embedding= WL_noconcate_fast(embedding,new_adj)
                #         embedding = preprocessing.normalize(embedding, norm='l2',axis=0, )
                    # if h>65:

                    # acc,nmi,f1,para,predict_labels = cmd.sc_linear(embedding,1,num_of_class,true_labels)
                    # time_end = time.perf_counter()  # 记录结束时间
                    # time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
                    # print(time_sum)
                    #
                    # tsne = TSNE(n_components=2,perplexity=55,learning_rate=0.00001,init='pca',n_iter=4000)
                    # node_features_tsne = tsne.fit_transform(node_features)
                    # tsne = TSNE(n_components=2,perplexity=55,learning_rate=0.01,init='pca',n_iter=4000)
                    # embedding_tsne = tsne.fit_transform(embedding)
                    # pplot2(node_features[pt], embedding[pt], f'h={h}', true_labels[pt], predict_labels[pt], p=800)
                    # data_points = embedding
                    # from scipy.spatial import distance
                    #
                    # distance_matrix = distance.cdist(data_points, data_points, metric='euclidean')
                    #
                    # # 计算平均距离
                    # average_similarity = np.sum(distance_matrix) / (len(data_points) * (len(data_points) - 1) / 2)
                    #
                    # print("平均距离:", average_similarity)
                    #
                    # print("平均相似度:", average_similarity)
                    # if average_similarity <0.01:
                    #     print(h)
                    #     break
                    # pplot2(node_features, embedding, f'h={h}', true_labels, true_labels, p=800)
                    # pplot_single(embedding_tsne,true_labels,name='vis_wl_{}'.format(h))
                    # print('@{} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset,h,para,acc,nmi,f1))

                    ##############################


                    #
                    # G = nx.from_numpy_matrix(adj_mat)
                    #
                    # pos = copy.deepcopy(embedding_tsne)
                    # # plt.figure(dpi=500)
                    #
                    #
                    # nx.draw_networkx_nodes(G, pos, node_size=3, node_color=true_labels)  # 画节点
                    # nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)  # 画边
                    # plt.show()



            #         if best_nmi < nmi:
            #             best_nmi = nmi
            #             best_h = h
            #         if best_f1 < f1:
            #             best_f1 = f1
            #         if best_acc < acc:
            #             best_acc = acc
            #         # print("True: ", true_labels)
            #
            #         # print("Pre: ", predict_labels)
            #
            #     # if best_nmi==1:
            #     #     print("Perfect!!!")
            #     #     break;
            #     print('@{} BEST(r= {} WL-{}) (h={}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset,r+1,para,best_h,best_acc,best_nmi,best_f1))
            #
            #     acc_li.append(best_acc)
            #     nmi_li.append(best_nmi)
            #     f1_li.append(best_f1)
            #
            # print('@{} BEST(rep= {} WL-{}) (h={}): ACC:{:.6f}({:.6f})  NMI:{:.6f}({:.6f})  f1_macro:{:.6f}({:.6f})'.format(dataset,rep, para, best_h, np.mean(acc_li),np.std(acc_li),np.mean(nmi_li),np.std(nmi_li), np.mean(f1_li),np.std(f1_li)))

    if emb_type['new_ikwl_noconcate'] == 1:
        # psili =[64,64,64,64,64,64,64,64,64,64,64,64,7,7,7,7,7,7]
        psili =[128]


        for dataset in datasets:
            adj_mat, node_features, true_labels = load_data(path1, dataset)
            myli = [i for i in range(len(node_features))]

            adj_mat = np.where(adj_mat!=0,1.0,0)
            new_adj = create_adj_avg(adj_mat)
            # G=nx.from_numpy_matrix(adj_mat)
            # adj_plot(adj_mat,true_labels,100)
            # adj_mat= sp.csr_matrix(adj_mat)
            num_of_class = np.unique(true_labels).shape[0]
            acc_li,nmi_li,f1_li=[],[],[]

            for r in range(rep):
                best_acc, best_nmi, best_f1, best_h, best_psi = -1, -1, -1, -1, -1
                for psi in psili:

                    embedding = node_features.copy()

                    time_start = time.perf_counter()
                    for h in range(1,4):

                        new_emb = np.where(embedding!=0,1,0)
                        embedding = IK_fm_dot(embedding, psi, t=200)
                        ans = np.sum(new_emb,axis=0)
                        ans.sort()
                        plt.plot(ans)
                        plt.show()
                        embedding = preprocessing.normalize(embedding, norm='l2',axis=0)
                        # re = random.sample(myli, 2000)
                        # re = sorted(re)
                        # i, j = np.ix_(re, re)
                        # adj = adj_mat[i, j]
                        # nfs = embedding[re]
                        # if h==0:
                        # def new_balanced(g, embedding):
                        #     new = np.zeros_like(embedding)
                        #     for ind in range(embedding.shape[0]):
                        #         tar = get_neigbors(g, ind)[1] + [ind]
                        #         new[ind] = np.mean(embedding[tar], axis=0)
                        #     return new
                        #     embedding2 =new_balanced(G,embedding)
                        #     embedding +=embedding2
                        # if h==0:
                        #     embedding2 =copy.deepcopy(embedding)
                        # else:
                        #     embedding2= np.concatenate([embedding,embedding2],axis=1)
                        #
                        # embedding = embedding2
                        # embedding = new_balanced(G, embedding)
                        # embedding1 = WL_max(embedding, adj_mat, G)

                        # embedding3 = WL_min(embedding, adj_mat, G)
                        # embedding = preprocessing.normalize(embedding, norm='l2',axis=0, )
                        # if h>65:
                        # embedding = np.concatenate([embedding1, embedding2, embedding3], axis=1)
                        # embedding =embedding2

                        embedding = WL_noconcate_fast(embedding, new_adj)
                        # emb=embedding


                        # sim = emb.dot(emb.T)
                        # np.fill_diagonal(sim,0)

                        # pt = []
                        # for i in range(node_features.shape[0]):
                        #     nei = get_neigbors(G, i, depth=1)[1]
                        #
                        #     if len(np.unique(true_labels[nei])) != 1:
                        #         pt.append(i)
                        #
                        emb = preprocessing.normalize(embedding, norm='l2')

                        acc, nmi, f1, para, predict_labels = cmd.sc_linear(emb, 1, num_of_class, true_labels)
                        # tsne = TSNE(n_components=2, perplexity=50,  init='pca',n_iter=4000,learning_rate=0.00001,)
                        # node_features_tsne = tsne.fit_transform(node_features)
                        tsne = TSNE(n_components=2, perplexity=30, learning_rate=0.01, init='pca', n_iter=3500)
                        embedding_tsne = tsne.fit_transform(embedding)
                        # pplot2(node_features_tsne[pt], embedding_tsne[pt], f'h={h}', true_labels[pt], predict_labels[pt], p=1000)
                        # pplot2(node_features_tsne, embedding_tsne, f'h={h}', true_labels, predict_labels, p=800)
                        pplot_single_seperate(embedding_tsne, true_labels,name=dataset)
                        pplot_single_seperate(embedding_tsne, predict_labels, name=dataset)
                        pplot_single(embedding_tsne, true_labels, name=dataset)
                        # # G = nx.from_numpy_matrix(adj_mat)
                        # # #
                        # # pos = embedding_tsne
                        # # # plt.figure(dpi=200)
                        # # #
                        # # nx.draw_networkx_nodes(G, pos, node_size=3, node_color=true_labels)  # 画节点
                        # # nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)  # 画边
                        # plt.show()
                        #
                        # emb = preprocessing.normalize(embedding, norm='l2',axis=1)
                        # acc, nmi, f1, para, predict_labels = cmd.sc_linear(emb, 1, num_of_class, true_labels)
                        time_end = time.perf_counter()  # 记录结束时间
                        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
                        # print(emb.shape, p," : ",time_sum)
                        print('@{} psi={} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset,psi,h,para,acc,nmi,f1))
                            # if best_nmi < nmi:
                        #     best_nmi = nmi
                        #     best_h = h
                        #     best_psi = psi
                        # if best_f1 < f1:
                        #     best_f1 = f1
                        # if best_acc < acc:
                        #     best_acc = acc


            #
            #     print('@BEST(r= {} IKWL-{}) (psi={} h={}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(r+1,para,best_psi,best_h,best_acc,best_nmi,best_f1))
            #     acc_li.append(best_acc)
            #     nmi_li.append(best_nmi)
            #     f1_li.append(best_f1)
            # print('@{} BEST(rep= {} IKWL-{}) (psi={} h={}): ACC:{:.6f}({:.6f})  NMI:{:.6f}({:.6f})  f1_macro:{:.6f}({:.6f})'.format(dataset,rep, para, best_psi, best_h, np.mean(acc_li),np.std(acc_li),np.mean(nmi_li),np.std(nmi_li), np.mean(f1_li),np.std(f1_li)))


    if emb_type['ikwl_noconcate'] == 1:
        psili =[4]
        for dataset in datasets:
            adj_mat, node_features, true_labels = load_data(path1, dataset)

            np.where(adj_mat != 0, adj_mat, 1)
            np.fill_diagonal(adj_mat, 0)
            num_of_class = np.unique(true_labels).shape[0]
            acc_li,nmi_li,f1_li=[],[],[]
            for r in range(rep):
                best_acc, best_nmi, best_f1, best_h, best_psi = -1, -1, -1, -1, -1
                new_adj = create_adj_avg(adj_mat)
                for psi in psili:
                    # embedding, new_map, dm, dm2 = lambda_feature_continous(node_features, node_features, eta=10, psi=psi, t=100)

                    embedding = IK_fm_dot(node_features, psi, t=100)
                    # embedding = node_features
                    for h in range(31):
                        if h > 0:
                            embedding = WL_noconcate_fast(embedding, new_adj)
                        # embedding = preprocessing.normalize(embedding, norm='l2',axis=0)

                        acc,nmi,f1,para,predict_labels = cmd.km(embedding,1,num_of_class,true_labels)
                        # tsne = TSNE(n_components=2, perplexity=5, learning_rate=0.00001, init='pca')
                        # node_features_tsne = tsne.fit_transform(node_features)
                        # tsne = TSNE(n_components=2, perplexity=50, learning_rate=0.0001, init='pca', n_iter=3000)
                        # embedding_tsne = tsne.fit_transform(embedding)
                        # pplot_single(embedding_tsne, true_labels, )

                        # G = nx.from_numpy_matrix(adj_mat)
                        # #
                        # pos = embedding_tsne
                        # # plt.figure(dpi=200)
                        # #
                        # nx.draw_networkx_nodes(G, pos, node_size=3, node_color=true_labels)  # 画节点
                        # nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)  # 画边
                        # plt.show()
                        print('@{} psi={} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset,psi,h,para,acc,nmi,f1))
            #             if best_nmi < nmi:
            #                 best_nmi = nmi
            #                 best_h = h
            #                 best_psi = psi
            #             if best_f1 < f1:
            #                 best_f1 = f1
            #             if best_acc < acc:
            #                 best_acc = acc
            #
            #
            #     print('@BEST(r= {} IKWL-{}) (psi={} h={}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(r+1,para,best_psi,best_h,best_acc,best_nmi,best_f1))
            #     acc_li.append(best_acc)
            #     nmi_li.append(best_nmi)
            #     f1_li.append(best_f1)
            # print('@{} BEST(rep= {} IKWL-{}) (psi={} h={}): ACC:{:.6f}({:.6f})  NMI:{:.6f}({:.6f})  f1_macro:{:.6f}({:.6f})'.format(dataset,rep, para, best_psi, best_h, np.mean(acc_li),np.std(acc_li),np.mean(nmi_li),np.std(nmi_li), np.mean(f1_li),np.std(f1_li)))

#@pubmed psi=64 h=24(sc_linear): ACC:0.705533  NMI:0.320438  f1_macro:0.697425


    if emb_type['new_gkwl_noconcate'] == 1:
        # psili =[64,64,64,64,64,64,64,64,64,64,64,64,7,7,7,7,7,7]
        psili = [4]
        gamma =[0.001]
        for dataset in datasets:
            adj_mat, node_features, true_labels = load_data(path1, dataset)
            adj_mat = np.where(adj_mat != 0, 1.0, 0)
            G = nx.from_numpy_matrix(adj_mat)
            # adj_mat= sp.csr_matrix(adj_mat)
            num_of_class = np.unique(true_labels).shape[0]
            acc_li, nmi_li, f1_li = [], [], []
            # gamma = [0.001 ]
            best_acc, best_nmi, best_f1, best_h, best_psi = -1, -1, -1, -1, -1
            for r in gamma:

                embedding = copy.deepcopy(node_features)
                new_adj = create_adj_avg(adj_mat)
                adj = copy.deepcopy(new_adj)
                from sklearn.kernel_approximation import Nystroem

                feature_map_nystroem = Nystroem(gamma=r, random_state=1, n_components=500)
                embedding = feature_map_nystroem.fit_transform(embedding)
                for h in range(25):
                    shape =  int(np.sqrt(adj_mat.shape[0]))

                    # embedding = preprocessing.normalize(embedding, norm='l2', axis=0)
                    #
                    embedding = WL_noconcate_fast(embedding, new_adj)
                    emb = preprocessing.normalize(embedding, norm='l2')
                    acc, nmi, f1, para, predict_labels = cmd.sc_linear(emb, 1, num_of_class, true_labels)
                    # print(embedding.shape)
                    # sim = emb.dot(emb.T)
                    # np.fill_diagonal(sim,0)

                    # tsne = TSNE(n_components=2, perplexity=5, learning_rate=0.00001, init='pca')
                    # node_features_tsne = tsne.fit_transform(node_features)
                    # tsne = TSNE(n_components=2, perplexity=30, learning_rate=0.01, init='pca', n_iter=3000)
                    # embedding_tsne = tsne.fit_transform(embedding)
                    # pplot2(node_features_tsne, embedding_tsne, f'h={h}', true_labels, predict_labels, p=1000)
                    # G = nx.from_numpy_matrix(adj_mat)
                    # #
                    # pos = embedding_tsne
                    # # plt.figure(dpi=200)
                    # #
                    # nx.draw_networkx_nodes(G, pos, node_size=3, node_color=true_labels)  # 画节点
                    # nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)  # 画边
                    # plt.show()
                    #
                    # time_end = time.perf_counter()  # 记录结束时间
                    # time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
                    # print(time_sum)
                    print('@{} psi={} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, gamma, h, para, acc, nmi, f1))
                    if best_nmi < nmi:
                        best_nmi = nmi
                        best_h = h

                    if best_f1 < f1:
                        best_f1 = f1
                    if best_acc < acc:
                        best_acc = acc
            print('@{} psi={} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset, gamma, best_h, para, best_acc, best_nmi,best_f1))

            # print('@{} BEST(rep= {} IKWL-{}) (psi={} h={}): ACC:{:.6f}({:.6f})  NMI:{:.6f}({:.6f})  f1_macro:{:.6f}({:.6f})'.format(dataset,rep, para, best_psi, best_h, np.mean(acc_li),np.std(acc_li),np.mean(nmi_li),np.std(nmi_li), np.mean(f1_li),np.std(f1_li)))
