import numpy as np
import copy

def hungarian_cluster_acc(x, y):
    assert x.shape == y.shape
    assert x.min() == 0
    assert y.min() == 0

    m = 1 + max(x.max(), y.max())
    n = len(x)
    total = np.zeros([m, m])
    for i in range(n):
        total[x[i], int(y[i])] += 1
    w = total.max() - total
    w = w - w.min(axis=1).reshape(-1, 1)
    w = w - w.min(axis=0).reshape(1, -1)
    while True:
        picked_axis0 = []
        picked_axis1 = []
        zerocnt = np.concatenate([(w == 0).sum(axis=1), (w == 0).sum(axis=0)], axis=0)

        while zerocnt.max() > 0:

            maxindex = zerocnt.argmax()
            if maxindex < m:
                picked_axis0.append(maxindex)
                zerocnt[np.argwhere(w[maxindex, :] == 0).squeeze(1) + m] = \
                    np.maximum(zerocnt[np.argwhere(w[maxindex, :] == 0).squeeze(1) + m] - 1, 0)
                zerocnt[maxindex] = 0
            else:
                picked_axis1.append(maxindex - m)
                zerocnt[np.argwhere(w[:, maxindex - m] == 0).squeeze(1)] = \
                    np.maximum(zerocnt[np.argwhere(w[:, maxindex - m] == 0).squeeze(1)] - 1, 0)
                zerocnt[maxindex] = 0
        if len(picked_axis0) + len(picked_axis1) < m:
            left_axis0 = list(set(list(range(m))) - set(list(picked_axis0)))
            left_axis1 = list(set(list(range(m))) - set(list(picked_axis1)))
            delta = w[left_axis0, :][:, left_axis1].min()
            w[left_axis0, :] -= delta
            w[:, picked_axis1] += delta
        else:
            break
    pos = []
    for i in range(m):
        pos.append(list(np.argwhere(w[i, :] == 0).squeeze(1)))

    def search(layer, path):

        # for i in pos:
        #     for j in i:
        #         if j not in path:
        #             path.append(j)
        # return path
        if len(path) == m:
            return path
        else:
            for i in pos[layer]:
                if i not in path:
                    newpath = copy.deepcopy(path)
                    newpath.append(i)
                    ans = search(layer + 1, newpath)
                    if ans is not None:
                        return ans

            return []

    path = search(0, [])

    totalcorrect = 0
    for i, j in enumerate(path):
        totalcorrect += total[i, j]

    return path,totalcorrect/n

def best_map(L1,L2):

    L1 = np.array(L1)
    L2 = np.array(L2)
    Label1 = np.unique(L1)       # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)        # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def err_rate(gt_s,s):
    gt_s = np.array(gt_s)
    s= np.array(s)
    print(gt_s)
    print(s)
    c_x=best_map(gt_s,s)
    err_x=np.sum(gt_s[:]!=c_x[:])
    missrate=err_x.astype(float)/(gt_s.shape[0])
    return missrate

def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    ri = (tp + tn) / (tp + tn + fp + fn)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    return f_beta

truth = np.array([0,0,0,0,1,1,1,2,2,2,2,2,3,3,3])
pred =  np.array([2,2,2,2,0,0,0,0,3,3,3,6,7,5,5])
print(hungarian_cluster_acc(truth,pred))

