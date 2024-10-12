import copy
import random
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from munkres import Munkres
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
import clustering_methods as cmd
from utils import GSNN, load_data,WL, WL_noconcate_one,WL_noconcate, IGK_WL_noconcate,IK_inne_fm,IK_fm_dot,WL_noconcate_gcn,pplot2,GSNN_try,WL_gao
from utils import create_adj_avg,WL_noconcate_fast,create_adj_avg_gcn,adj_plot
import warnings
import scipy.io as sio
from sklearn import preprocessing
from tqdm import tqdm
from utils import load_data_noise
import os

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

def expansion_rate(ind,g,adj_mat):

    deg = np.sum(adj_mat, axis=1)
    tar = get_neigbors(g,ind)[1]
    tar.append(ind)
    a1 = np.sum(deg[tar])
    new_tar = []
    for i in tar:
        new_tar.append(i)
        temp = get_neigbors(g,i)[1]
        new_tar.extend(temp)
    new_tar =list(set(new_tar))
    a2 = np.sum(deg[new_tar])

    s= a1/a2
    return s

def group_partition(predict_labels):
## 获取每个group的下标
    num_of_class = np.unique(predict_labels).shape[0]
    group_of_pre = []
    for i in range(num_of_class):
        temp = np.where(predict_labels == i)[0].tolist()
        group_of_pre.append(temp)
    return group_of_pre

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

def new_balanced(g,embedding):
        new=np.zeros_like(embedding)
        for ind in range(embedding.shape[0]):

            tar = get_neigbors(g, ind)[1]+[ind]
            new[ind] = np.mean(embedding[tar],axis=0)
        return new

def noise_exp(dataset='cora',psi_IK=32,h_=5,interclass_noise_rate=0.1,intraclass_noise_rate=0.1):
    path1 = 'E:/Graph Clustering/dataset/real_world data/'
    # path1 = 'E:/Graph Clustering/dataset/artificial data/'
    # dataset = 'cora'
    # dataset = 'Graph_imbalanced'
    # adj_mat, node_features, true_labels = load_data(dataset)
    adj_mat, node_features, true_labels = load_data_noise(dataset,interclass_noise_rate=interclass_noise_rate,intraclass_noise_rate=intraclass_noise_rate)
    temp_li=[]
    # for i,j in enumerate(true_labels):
    #     if j==2 or j ==4:
    #         temp_li.append(i)
    # i, j = np.ix_(temp_li,temp_li)
    #
    # adj_mat=adj_mat[i,j]
    # node_features=node_features[temp_li]
    # true_labels=true_labels[temp_li]


    # y=[0 for i in range(adj_mat.shape[0])]
    G = nx.from_numpy_matrix(adj_mat)

    num_of_class = np.unique(true_labels).shape[0]

    # ind1=7
    # ind2=8
    # ind3=9
    # for i in p_li:
    #     y[i]=2
    #     y[i+10],y[i+11],y[i+12]=2,2,2

    tsne = TSNE(n_components=2,perplexity=40,n_iter=5000)
    node_features_tsne = tsne.fit_transform(node_features)

    deg = np.sum(adj_mat, axis=1)


    # pos = node_features_tsne
    # nx.draw_networkx_nodes(G, pos, node_size=2, node_color=true_labels)  # 画节点
    # nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.2)  # 画边
    # plt.show()

    # res1 = expansion_rate(ind1,G,adj_mat)
    # res2 = expansion_rate(ind2,G,adj_mat)
    # deg1 = deg[ind1]
    # deg2 = deg[ind2]
    # print(res1,res2)

    embedding = node_features.copy()

    new_adj = create_adj_avg(adj_mat)


    #####del
    # def new_balanced(g,embedding):
    #     new=np.zeros_like(embedding)
    #     for ind in range(embedding.shape[0]):

    #         tar = get_neigbors(g, ind)[1]+[ind]
    #         new[ind] = np.mean(embedding[tar],axis=0)
    #     return new
    # adj_plot(adj_mat,true_labels,scale=50)
    psi=psi_IK
    adj = copy.deepcopy(adj_mat)

    # acc_final, nmi_final = 0,0
    # f1_final, para_final = 0,-1 
    # predict_labels_final = -1

    acc_lst,nmi_lst = [],[]
    f1_lst = []
    # predict_labels_lst = []

    for h in range(h_):

        embedding = IK_fm_dot(embedding, psi=psi, t=200)
        embedding = preprocessing.normalize(embedding, norm='l2', axis=0)

        embedding = WL_noconcate_fast(embedding, new_adj)
        emb =preprocessing.normalize(embedding, norm='l2')

        li=[i for i in range(adj_mat.shape[0])]
        rs = random.sample(li,200)
        embedding2=emb[rs]
        true_labels2=true_labels[rs]
        if h==0:
            acc, nmi, f1, para, predict_labels2 = cmd.sc_linear(embedding2, 1, num_of_class, true_labels2)
        else:
            predict_labels2=np.array(new_predict_labels)[rs]

        group = group_partition(predict_labels2)
        group_mean= np.zeros((num_of_class,embedding2.shape[1]))
        for i in range(len(group)):
            group_mean[i] = np.mean(embedding2[group[i]],axis=0)
        # sim = np.zeros_like(adj_mat,dtype=float)
        # for i in range(embedding.shape[0]):
        #     ind = predict_labels[i]
        #     sim[i] = embedding[i].dot(group_mean[ind].T)
        #
        sim = emb.dot(group_mean.T)

        # for i in range(sim.shape[0]):
        #     sim[i] = sim[i]/np.sum(sim[i])
        import clustering_metric as cm
        predict_labels = [np.argmax(sim[i]) for i in range(sim.shape[0])]
        translate, new_predict_labels = cm.translate(true_labels, predict_labels)


        for i in range(adj_mat.shape[0]):
            for j in range(adj_mat.shape[1]):
                # indi = np.argmax(sim[i])
                # indj = np.argmax(sim[j])
                # if indi==indj:
                #     s=1
                # else:
                #     s=0.001
                # max1 = np.sort(sim[j])[-1]
                # max2 = np.sort(sim[j])[-2]
                # t=(max1-max2)/max1
                adj_mat[i][j]= sim[i].dot(sim[j].T)*np.max(sim[i])

        adj_mat = adj*adj_mat
        new_adj = create_adj_avg(adj_mat)

        acc, nmi, f1, para, predict_labels = cmd.sc_linear(emb, 1, num_of_class, true_labels)
        acc_lst.append(acc)
        nmi_lst.append(nmi)
        f1_lst.append(f1)

        print('@h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(h, para, acc, nmi, f1))

    # Save Results  
    save_folder = 'mWDK_noise_effect_results'
    file_name = dataset+'_psi_'+str(psi)+'_h_'+str(h_)+'_inter_'+str(interclass_noise_rate)+'_intra_'+str(intraclass_noise_rate)+'.mat'
    saved_data = {'nmi':np.array(nmi_lst), 'acc':np.array(acc_lst), 'f1':np.array(f1_lst)}
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sio.savemat(save_folder+os.sep+file_name, saved_data)
    print('Data saved in {}.'.format(save_folder+os.sep+file_name))

if __name__ == '__main__':
    psi_lst = [2,4,6,8,16,32,64,128]
    # Remember to change the specfic h in each dataset!
    interclass_noise_rate_lst = [0,0.1/16,0.1/8,0.1/4,0.1/2,0.1]
    intraclass_noise_rate_lst = [0,0.1,0.2,0.4]
    for psi in psi_lst:
        for interclass_noise_rate in interclass_noise_rate_lst:
            for intraclass_noise_rate in intraclass_noise_rate_lst:
                print('psi:',psi,'interclass_noise_rate:',interclass_noise_rate,'intraclass_noise_rate:',intraclass_noise_rate)
                noise_exp(dataset='dblp',psi_IK=psi,h_=5,
                          interclass_noise_rate=interclass_noise_rate,
                          intraclass_noise_rate=intraclass_noise_rate)


