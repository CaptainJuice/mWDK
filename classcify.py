import sklearn
from sklearn.model_selection import train_test_split
import copy
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, normalized_mutual_info_score, accuracy_score, classification_report
from munkres import Munkres
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
import clustering_methods as cmd
from utils import GSNN, load_data,WL, WL_noconcate_one,WL_noconcate, IGK_WL_noconcate,IK_inne_fm,IK_fm_dot,WL_noconcate_gcn,pplot2,pplot3,pplot_single
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

if __name__ == '__main__':
    ###################################### parm ########################################
    emb_type = {"wl_noconcate":21,
              "ikwl_noconcate":12,
              "new_ikwl_noconcate":1,
            "new_gkwl_noconcate": 12
               }
    path1 = 'E:/Graph Clustering/dataset/real_world data/'
    datasets1 = ['cora','citeseer','pubmed','amap',]
    datasets2 = ['blogcatalog','flickr','wiki','dblp','acm']
    # path1 = 'E:/Graph Clustering/dataset/artificial data/'
    datasets = ['cora']
    # datasets = ['ENodes_UDegrees']
    ####################################################################################

    rep =1
    if emb_type['wl_noconcate'] == 1:#312
        for dataset in datasets:

            adj_mat, node_features, true_labels = load_data(path1, dataset)
            G=nx.from_numpy_matrix(adj_mat)
            adj_plot(adj_mat,true_labels,100)

            adj2=copy.deepcopy(adj_mat)
            num_of_class = np.unique(true_labels).shape[0]
            np.where(adj_mat != 0, adj_mat, 1)


            for r in range(rep):
                emb = node_features.copy()
                time_start = time.perf_counter()
                embedding = node_features.copy()

                new_adj = create_adj_avg(adj_mat)
                pt = []
                for i in range(node_features.shape[0]):
                    nei = get_neigbors(G,i,depth=1)[1]

                    if len(np.unique(true_labels[nei]))!=1:
                        pt.append(i)
                print(len(pt))
                for h in range(15):
                    if h >0:
                        embedding = WL_noconcate_fast(embedding,new_adj)
                    data = np.hstack((embedding, true_labels.reshape(2708, 1)))
                    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
                    from sklearn.model_selection import GridSearchCV

                    X_train, y_train = train_set[:, :-1], train_set[:, -1]
                    X_test, y_test = test_set[:, :-1], test_set[:, -1]
                    knn = sklearn.neighbors.KNeighborsClassifier()
                    grid = {"n_neighbors": range(1, 11, 1), "weights": ['uniform', 'distance']}
                    gs = GridSearchCV(estimator=knn,
                                      param_grid=grid,
                                      scoring="accuracy",
                                      n_jobs=-1,
                                      cv=5,
                                      verbose=0,
                                      )
                    gs.fit(X_train, y_train)
                    # 最好的分值。
                    # print(gs.best_score_)
                    # 最好的超参数组合。
                    # print(gs.best_params_)
                    # 使用最好的超参数训练好的模型。
                    # print(gs.best_estimator_)
                    estimator = gs.best_estimator_
                    y_hat = estimator.predict(X_test)
                    # print(classification_report(y_test, y_hat))
                    print(estimator.score(X_test, y_test))
                    # acc,nmi,f1,para,predict_labels = cmd.km(embedding,1,num_of_class,true_labels)
                    # # time_end = time.perf_counter()  # 记录结束时间
                    # # time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
                    # # print(time_sum)
                    # #
                    # tsne = TSNE(n_components=2,perplexity=55,learning_rate=0.00001,init='pca',n_iter=4000)
                    # node_features_tsne = tsne.fit_transform(node_features)
                    # tsne = TSNE(n_components=2,perplexity=55,learning_rate=0.00001,init='pca',n_iter=4000)
                    # embedding_tsne = tsne.fit_transform(embedding)
                    # # # pplot2(node_features[pt], embedding[pt], f'h={h}', true_labels[pt], predict_labels[pt], p=800)
                    # # # pplot2(node_features_tsne, embedding_tsne, f'h={h}', true_labels, predict_labels, p=800)
                    # pplot_single(embedding_tsne,true_labels,)
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
        psili =[64,128]*100
        for dataset in datasets:
            adj_mat, node_features, true_labels = load_data(path1, dataset)
            adj_mat = np.where(adj_mat!=0,1.0,0)
            G=nx.from_numpy_matrix(adj_mat)
            adj_plot(adj_mat,true_labels,100)
            # adj_mat= sp.csr_matrix(adj_mat)
            num_of_class = np.unique(true_labels).shape[0]
            acc_li,nmi_li,f1_li=[],[],[]

            for r in range(rep):
                best_acc, best_nmi, best_f1, best_h, best_psi = -1, -1, -1, -1, -1
                for psi in psili:
                    embedding = node_features.copy()

                    # time_start = time.perf_counter()
                    # aa = adj_mat.dot(adj_mat)
                    # aa = np.where(aa>0,0.5,0)
                    # adj_mat += aa
                    new_adj = create_adj_avg(adj_mat)
                    adj = copy.deepcopy(new_adj)
                    for h in range(31):

                        embedding = IK_fm_dot(embedding, psili[h], t=100)
                        embedding = preprocessing.normalize(embedding, norm='l2',axis=0)
                        # if h==0:
                        def new_balanced(g, embedding):
                            new = np.zeros_like(embedding)
                            for ind in range(embedding.shape[0]):
                                tar = get_neigbors(g, ind)[1] + [ind]
                                new[ind] = np.mean(embedding[tar], axis=0)
                            return new
                        #     embedding2 =new_balanced(G,embedding)
                        #     embedding +=embedding2
                        # if h==0:
                        #     embedding2 =copy.deepcopy(embedding)
                        # else:
                        #     embedding2= np.concatenate([embedding,embedding2],axis=1)
                        #
                        # embedding = new_balanced(G, embedding)
                        embedding = WL_noconcate_fast(embedding, new_adj)
                        embedding = IK_fm_dot(embedding, 16, t=100)
                        emb = preprocessing.normalize(embedding, norm='l2')
                        data = np.hstack((emb, true_labels.reshape(2708, 1)))
                        train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
                        from sklearn.model_selection import GridSearchCV

                        X_train, y_train = train_set[:, :-1], train_set[:, -1]
                        X_test, y_test = test_set[:, :-1], test_set[:, -1]
                        knn = sklearn.neighbors.KNeighborsClassifier()
                        # 定义需要尝试的超参数组合。
                        grid = {"n_neighbors": range(1, 20, 1), "weights": ['uniform', 'distance']}
                        # estimator：评估器，即对哪个模型调整超参数。
                        # param_grid：需要检验的超参数组合。从这些组合中，寻找效果最好的超参数组合。
                        # scoring：模型评估标准。
                        # n_jobs：并发数量。
                        # cv：交叉验证折数。
                        # verbose：输出冗余信息，值越大，输出的信息越多。
                        gs = GridSearchCV(estimator=knn,
                                          param_grid=grid,
                                          scoring="accuracy",
                                          n_jobs=-1,
                                          cv=5,
                                          verbose=0,
                                          )
                        gs.fit(X_train, y_train)
                        # 最好的分值。
                        # print(gs.best_score_)
                        # # 最好的超参数组合。
                        # print(gs.best_params_)
                        # # 使用最好的超参数训练好的模型。
                        # print(gs.best_estimator_)
                        estimator = gs.best_estimator_
                        y_hat = estimator.predict(X_test)
                        # print(classification_report(y_test, y_hat))
                        print(estimator.score(X_test, y_test))
                        # acc_li, nmi_li, f1_li = [], [], []
                        #
                        # # tsne = TSNE(n_components=2, perplexity=5,  init='pca',n_iter=4000,learning_rate=0.00001,)
                        # # node_features_tsne = tsne.fit_transform(node_features)
                        # # tsne = TSNE(n_components=2, perplexity=55,init='pca',n_iter=4000,  learning_rate=0.001, )
                        # # embedding_tsne = tsne.fit_transform(embedding)
                        # # # pplot2(node_features_tsne[pt], embedding_tsne[pt], f'h={h}', true_labels[pt], predict_labels[pt], p=1000)
                        # # # pplot2(node_features_tsne, embedding_tsne, f'h={h}', true_labels, predict_labels, p=800)
                        # # pplot_single(embedding_tsne, true_labels,)
                        # # #
                        # # # # G = nx.from_numpy_matrix(adj_mat)
                        # # # # #
                        # # # # pos = embedding_tsne
                        # # # # # plt.figure(dpi=200)
                        # # # # #
                        # # # # nx.draw_networkx_nodes(G, pos, node_size=3, node_color=true_labels)  # 画节点
                        # # # # nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)  # 画边
                        # # plt.show()
                        # #
                        # # time_end = time.perf_counter()  # 记录结束时间
                        # # time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
                        # # print(time_sum)
                        # print('@{} psi={} h={}({}): ACC:{:.6f}  NMI:{:.6f}  f1_macro:{:.6f}'.format(dataset,psi,h,para,acc,nmi,f1))
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
        psili =[32]
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
        gamma =0.01
        for dataset in datasets:
            adj_mat, node_features, true_labels = load_data(path1, dataset)
            adj_mat = np.where(adj_mat != 0, 1.0, 0)
            G = nx.from_numpy_matrix(adj_mat)
            # adj_mat= sp.csr_matrix(adj_mat)
            num_of_class = np.unique(true_labels).shape[0]
            acc_li, nmi_li, f1_li = [], [], []
            gamma = [0.001,0.001,0.0001,0.0001,10,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,]
            best_acc, best_nmi, best_f1, best_h, best_psi = -1, -1, -1, -1, -1
            for r in gamma:

                embedding = copy.deepcopy(node_features)
                new_adj = create_adj_avg(adj_mat)
                adj = copy.deepcopy(new_adj)
                for h in range(25):
                    shape =  int(np.sqrt(adj_mat.shape[0]))
                    from sklearn.kernel_approximation import Nystroem

                    feature_map_nystroem = Nystroem(gamma=gamma[h], random_state=1, n_components=500)
                    embedding= feature_map_nystroem.fit_transform(embedding)
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
