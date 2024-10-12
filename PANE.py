import numpy as np
from sklearn.cluster import KMeans as km
# data = np.fromfile('emb/cora.128.a.bin.b',dtype=np.float).reshape(2708,-1)
import clustering_methods as cmd
# print(data)
from utils import WL_noconcate, load_data,WL,IK_fm_dot
from sklearn import preprocessing
import sklearn.svm as svm
import sklearn

path1 = 'E:/Graph Clustering/dataset/real_world data/'

datasets = ['cora' ,'citeseer', 'wiki', 'dblp', 'acm',]
# datasets =['pubmed','amap',"flickr","amac","blogcatalog"]
datasets = ['cora', 'citeseer', 'wiki', 'acm', 'dblp', 'amap','pubmed','pubmed','blogcatalog','flickr' ]
datasets= ['flickr']
path1 = 'E:/Graph Clustering/dataset/artificial data/'
# datasets = ['cora']
datasets = ['ENodes_EDegrees_Easy']
# datasets=["Graph_1","Graph_2","Graph_3",]
# true_labels = np.load('C:/Users/Admin/Desktop/PANE-main/algos/pane/label.npy')
for data in datasets:
    print("==========================={}==========================".format(data))
    adj_mat, node_features, true_labels = load_data(path1, data)
    num_of_class = np.unique(true_labels).shape[0]

    Xf = np.load('C:/Users/Admin/Desktop/PANE-main/algos\pane/{}_xf.npy'.format(data))
    Xb= np.load('C:/Users/Admin/Desktop/PANE-main/algos\pane/{}_xb.npy'.format(data))
    # Xf = preprocessing.normalize(Xf, norm='l2', axis=1)
    # Xb = preprocessing.normalize(Xb, norm='l2', axis=1)
    # print(np.isinf(train).any()[np.isinf(train).any() == True])
    features = np.hstack([Xf, Xb])
    # # features=Xf
    # # features= np.concatenate((data1,data2),axis=1)
    # # features = Xb
    acc_li, nmi_li, f1_li = [], [], []

    for i in range(7):
        acc,nmi,f1,para,predict_labels = cmd.sc_linear(features,1,num_of_class,true_labels)
        acc_li.append(acc)
        nmi_li.append(nmi)
        f1_li.append(f1)
    print('@Both({}): ACC:{:.6f}({:.6f})  NMI:{:.6f}({:.6f})  f1_macro:{:.6f}({:.6f})'.format(para,np.mean(acc_li),np.std(acc_li),np.mean(nmi_li),np.std(nmi_li),np.mean(f1_li),np.std(f1_li),))
    #
    # features = Xf

    # minMaxScaler = preprocessing.MinMaxScaler().fit(Xb)
    # Xb = minMaxScaler.transform(Xb)
    features = Xb
    acc_li, nmi_li, f1_li = [], [], []
    for i in range(7):
        acc,nmi,f1,para,predict_labels = cmd.sc_linear(features,1,num_of_class,true_labels)
        acc_li.append(acc)
        nmi_li.append(nmi)
        f1_li.append(f1)
    print('@xb({}): ACC:{:.6f}({:.6f})  NMI:{:.6f}({:.6f})  f1_macro:{:.6f}({:.6f})'.format(para,np.mean(acc_li),np.std(acc_li),np.mean(nmi_li),np.std(nmi_li),np.mean(f1_li),np.std(f1_li),))

    #
    # features = Xf
    # for i in range(10):
    #     acc,nmi,f1,para,predict_labels = cmd.sc_linear(features,1,num_of_class,true_labels)
    #     acc_li.append(acc)
    #     nmi_li.append(nmi)
    #     f1_li.append(f1)
    # print('@xf({}): ACC:{:.6f}({:.6f})  NMI:{:.6f}({:.6f})  f1_macro:{:.6f}({:.6f})'.format(para,np.mean(acc_li),np.std(acc_li),np.mean(nmi_li),np.std(nmi_li),np.mean(f1_li),np.std(f1_li),))