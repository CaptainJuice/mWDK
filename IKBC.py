import numpy as np
import networkx as nx
import random
import scipy.io as sio
from sklearn import preprocessing
import clustering_metric as cm
import warnings
warnings.filterwarnings('ignore')
"""
    You can conveniently use 'auto_search_tau' to get a list of predict labels for each tau by brute force search(Recommended).
    If you need the clustering result given a certain tau, please call the function 'ikbc'/'lkbc' directly.
"""


def auto_search_tau(node_features,n_cluster,kernel='ik',psi=None,t=None):
    """
        paras: kernel can be 'ik' or 'lk'(linear kernel). Parameters psi and t are only for 'ik'.
        func: auto-search tau from range [0.05,0.10,0.15,···,0.95]
        return: a list of predict labels for each tau
    """
    result = []
    if kernel == 'ik':
        ### feature mapping ###
        features_mapping = IK_inne_fm(node_features, psi=psi, t=t)
    elif kernel == 'lk':
        features_mapping = node_features
    for tt in range(5,100,5):
        tau = tt * 0.01
        ### sample ###
        sample_num = min(features_mapping.shape[0], 1000)  # sample size
        node_id = [i for i in range(features_mapping.shape[0])]
        sample_id = random.sample(node_id, sample_num)
        sample_id.sort()
        sample_id = np.array(sample_id)
        sample_features = features_mapping[sample_id, :]

        ### similarity matrix ###
        if kernel == 'ik':
            similarity_mat = np.dot(sample_features, sample_features.T) / t
        elif kernel == 'lk':
            similarity_mat = np.dot(sample_features, sample_features.T)
            similarity_mat = similarity_mat / np.max(similarity_mat)
        adj_mat = (similarity_mat >= tau).astype(int)
        G = nx.from_numpy_matrix(adj_mat)
        connection = [np.array(list(i)) for i in nx.connected_components(G)]
        connection = [sample_id[i] for i in connection]
        len_li = [len(i) for i in connection]
        sorted_id = sorted(range(len(len_li)), key=lambda k: len_li[k], reverse=True)[:n_cluster]

        clusters = [connection[id] for id in sorted_id]
        tar_id = [j for i in clusters for j in i]
        tar_embedding = np.array([np.mean(features_mapping[ids], axis=0).tolist() for ids in clusters])
        rest_id = [x for x in node_id if x not in tar_id]
        rest_embedding = features_mapping[rest_id, :]
        similarity = np.dot(rest_embedding, tar_embedding.T).tolist()

        for i in range(len(similarity)):
            label_id = similarity[i].index(max(similarity[i]))
            clusters[label_id] = np.append(clusters[label_id], rest_id[i])

        predict_labels = np.zeros(len(node_id), dtype=int)
        for i, element in enumerate(clusters):
            predict_labels[np.array(element)] = i
        ### postprocessing ###
        flag = np.ceil(len(node_id) * 0.01)
        for _ in range(100):
            post_mean = []
            for i in range(n_cluster):
                temp_mean = np.mean(features_mapping[predict_labels == i], axis=0)
                post_mean.append(temp_mean)
            post_mean = np.array(post_mean)
            post_labels = np.zeros(len(node_id), dtype=int)
            post_similarity = np.dot(features_mapping, post_mean.T).tolist()
            for i in range(len(post_similarity)):
                post_id = post_similarity[i].index(max(post_similarity[i]))
                post_labels[i] = post_id

            if sum(predict_labels != post_labels) <= flag:
                break
            predict_labels = post_labels

        result.append(predict_labels)
    return result

def gaussian_kernel_3(X, Y, sigma):
    '''
    输入
        X       二维浮点数组, 第一维长度num1是X样本数量, 第二维长度dim是特征长度
        X       二维浮点数组, 第一维长度num2是Y样本数量, 第二维长度dim是特征长度
        sigma   浮点数, 高斯核的超参数
    输出
        K       二位浮点数组, 第一维长度是num1, 第二维长度是num2
    '''
    X = np.array(X)
    Y = np.array(Y)
    D2 = np.sum(X*X, axis=1, keepdims=True) \
            + np.sum(Y*Y, axis=1, keepdims=True).T \
            - 2 * np.dot(X, Y.T)
    return np.exp(-D2 / (2 * sigma ** 2))




def ik_bc(node_features,n_cluster,psi,t,tau):
    """
        func: ikbc given psi and tau
        return: predict labels
    """
    ### feature mapping using isolation kernel(iNNE) ###
    features_mapping = IK_inne_fm(node_features,psi=psi,t=t)
    ### sample ###
    sample_num = min(node_features.shape[0],1000)  # sample size
    node_id = [i for i in range(node_features.shape[0])]
    sample_id = random.sample(node_id,sample_num)
    sample_id.sort()
    sample_id = np.array(sample_id)
    sample_features = features_mapping[sample_id,:]

    ### similarity matrix ###
    similarity_mat = np.dot(sample_features,sample_features.T)/t
    adj_mat = (similarity_mat >= tau).astype(int)
    G = nx.from_numpy_matrix(adj_mat)
    connection = [np.array(list(i)) for i in nx.connected_components(G)]
    connection = [sample_id[i] for i in connection]
    len_li = [len(i) for i in connection]
    sorted_id = sorted(range(len(len_li)), key=lambda k: len_li[k], reverse=True)[:n_cluster]

    clusters = [connection[id] for id in sorted_id]
    tar_id = [j for i in clusters for j in i]
    tar_embedding = np.array([np.mean(features_mapping[ids], axis=0).tolist() for ids in clusters])
    rest_id = [x for x in node_id if x not in tar_id]
    rest_embedding = features_mapping[rest_id,:]
    similarity = np.dot(rest_embedding, tar_embedding.T).tolist()

    for i in range(len(similarity)):
        label_id = similarity[i].index(max(similarity[i]))
        clusters[label_id] = np.append(clusters[label_id],rest_id[i])

    predict_labels = np.zeros(len(node_id), dtype=int)
    for i, element in enumerate(clusters):
        predict_labels[np.array(element)] = i
    ### postprocessing ###
    flag = np.ceil(len(node_id)*0.01)
    for _ in range(100):
        post_mean =[]
        for i in range(n_cluster):
            temp_mean = np.mean(features_mapping[predict_labels==i],axis=0)
            post_mean.append(temp_mean)
        post_mean = np.array(post_mean)
        post_labels = np.zeros(len(node_id),dtype=int)
        post_similarity = np.dot(features_mapping,post_mean.T).tolist()
        for i in range(len(post_similarity)):
            post_id = post_similarity[i].index(max(post_similarity[i]))
            post_labels[i] = post_id

        if sum(predict_labels!=post_labels) <= flag:
            break
        predict_labels = post_labels
    return predict_labels
def lkbc(node_features,n_cluster,tau):
    """
        func: linear kernel bounded clustering
        return: predict labels
    """
    node_id = [i for i in range(node_features.shape[0])]

    # node_features = preprocessing.scale(node_features)
    node_features = preprocessing.normalize(node_features,'l2')
    ### no feature mapping ###
    features_mapping = node_features
    ### sample ###


    sample_num = min(node_features.shape[0],5000)  # sample size
    sample_id = random.sample(node_id,sample_num)
    sample_id.sort()
    sample_id = np.array(sample_id)
    sample_features = features_mapping[sample_id,:]

    ### similarity matrix ###
    similarity_mat = np.dot(sample_features,sample_features.T)
    # similarity_mat = preprocessing.scale(similarity_mat)
    # similarity_mat = np.where(similarity_mat < 0, 0, similarity_mat)

    # similarity_mat = np.dot(sample_features,sample_features.T)
    # similarity_mat = similarity_mat/np.max(similarity_mat)
    # similarity_mat = gaussian_kernel_3(sample_features,sample_features,0.001)
    # from sklearn.metrics.pairwise import cosine_similarity
    # similarity_mat = cosine_similarity(sample_features)
    # from scipy.spatial.distance import pdist, squareform
    # Y = pdist(sample_features, 'cosine')
    # similarity_mat = squareform(Y)
    # similarity_mat = 1-similarity_mat
    # np.fill_diagonal(similarity_mat,0.5)

    # similarity_mat = similarity_mat / np.max(similarity_mat)
    # similarity_mat = preprocessing.normalize(similarity_mat,"l2")
    # np.fill_diagonal(similarity_mat,1)

    similarity_mat = (similarity_mat+similarity_mat.T)*0.5
    adj_mat = (similarity_mat >= tau).astype(int)
    G = nx.from_numpy_matrix(adj_mat)
    connection = [np.array(list(i)) for i in nx.connected_components(G)]
    connection = [sample_id[i] for i in connection]
    len_li = [len(i) for i in connection]
    sorted_id = sorted(range(len(len_li)), key=lambda k: len_li[k], reverse=True)[:n_cluster]

    clusters = [connection[id] for id in sorted_id]
    tar_id = [j for i in clusters for j in i]
    tar_embedding = np.array([np.mean(features_mapping[ids], axis=0).tolist() for ids in clusters])
    rest_id = [x for x in node_id if x not in tar_id]
    rest_embedding = features_mapping[rest_id,:]
    similarity = np.dot(rest_embedding, tar_embedding.T).tolist()

    for i in range(len(similarity)):
        label_id = similarity[i].index(max(similarity[i]))
        clusters[label_id] = np.append(clusters[label_id],rest_id[i])

    predict_labels = np.zeros(len(node_id), dtype=int)
    for i, element in enumerate(clusters):
        predict_labels[np.array(element)] = i
    # postprocessing
    flag = np.ceil(len(node_id)*0.01)
    for _ in range(100):

        post_mean =[]
        for i in range(n_cluster):
            temp_mean = np.mean(features_mapping[predict_labels==i],axis=0)
            post_mean.append(temp_mean)
        post_mean = np.array(post_mean)
        post_labels = np.zeros(len(node_id),dtype=int)

        post_similarity = np.dot(features_mapping,post_mean.T).tolist()
        for i in range(len(post_similarity)):
            post_id = post_similarity[i].index(max(post_similarity[i]))
            post_labels[i] = post_id

        if sum(predict_labels!=post_labels) <= flag:
            break
        predict_labels = post_labels

    return predict_labels


def IK_inne_fm(X,psi,t=100):

    onepoint_matrix = np.zeros((X.shape[0], (int)(t*psi)), dtype=int)
    for time in range(t):
        sample_num = psi  #
        sample_list = [p for p in range(len(X))]
        sample_list = random.sample(sample_list, sample_num)
        sample = X[sample_list, :]

        tem1 = np.dot(np.square(X), np.ones(sample.T.shape))  # n*psi
        tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)  # n*psi

        tem = np.dot(np.square(sample), np.ones(sample.T.shape))
        sample2sample = tem + tem.T - 2 * np.dot(sample, sample.T)
        sample2sample[sample2sample < 1e-9] = 99999999;
        radius_list = np.min(sample2sample, axis=1)

        min_point2sample_index=np.argmin(point2sample, axis=1)
        min_dist_point2sample = min_point2sample_index+time*psi
        point2sample_value = point2sample[range(len(onepoint_matrix)), min_point2sample_index]
        ind=point2sample_value < radius_list[min_point2sample_index]
        onepoint_matrix[ind,min_dist_point2sample[ind]]=1
    return onepoint_matrix

if __name__ == '__main__':
    data = sio.loadmat('E:\Graph Clustering\Matlab\IKBC_base\IKBC\spiral.mat')
    node_features = data['data']
    true_labels = data['class'].reshape(-1).tolist()

    num_of_class = np.unique(true_labels).shape[0]
    psi_li = [2,4,6,8,16,24,48,64,80,100,200,250,512]
    t=100
    for psi in psi_li:
        result = auto_search_tau(node_features,n_cluster=num_of_class,kernel='ik',psi=psi,t=t)
        nmi_li, acc_li,f1_li=[],[],[]
        for predict_labels in result:
            translate, new_predict_labels = cm.translate(true_labels, predict_labels)
            acc, nmi, f1, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, adjscore = cm.evaluationClusterModelFromLabel(true_labels, new_predict_labels)
            nmi_li.append(nmi)
            f1_li.append(f1)
            acc_li.append(acc)
        print("@psi={} Acc:{}  NMI:{}  f1-macro: {}".format(psi,max(acc_li),max(nmi_li),max(f1_li)))