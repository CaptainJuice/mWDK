import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DensityPeak:
    def __init__(self,X,n_cluster):
        self.X=X
        self.n_cluster=n_cluster
        distance_matrix = pairwise_distances(X)
        distance_matrix = distance_matrix / np.max(distance_matrix)
        self.distance_matrix = np.around(distance_matrix, 3)

    def autoSearch(self):

        epsilon_list = np.unique(self.distance_matrix)
        P = sum(epsilon_list < 1)
        result_list = np.zeros((P - 1, len(self.X)),dtype=int)
        centers_list = np.zeros((P - 1, self.n_cluster), dtype=int)
        for i in range(1, P):
            predict_label,center_index = self.DP(epsilon=epsilon_list[i])
            result_list[i - 1, :] = predict_label
            centers_list[i-1,:]=center_index

        return result_list,self.X[centers_list]

    def DP(self,epsilon):
        density = sum(self.distance_matrix.T <= epsilon)
        indexListSortedByDensity = np.argsort(-density)
        delta = np.zeros(len(self.distance_matrix))
        delta[indexListSortedByDensity[0]] = 1.00001
        neighborIndexList = np.zeros(len(self.distance_matrix), dtype=int)
        # cal delta
        for i, sampleIndex in enumerate(indexListSortedByDensity):
            if i == 0:
                continue
            candidate_list = indexListSortedByDensity[:i]
            index = np.argmin(self.distance_matrix[sampleIndex, candidate_list])
            neighborIndexList[sampleIndex] = candidate_list[index]
            delta[sampleIndex] = self.distance_matrix[sampleIndex][neighborIndexList[sampleIndex]]
        # normalize
        density = MinMaxScaler().fit_transform(density.reshape(-1, 1)).reshape(-1) + 0.0000001;
        delta = MinMaxScaler().fit_transform(delta.reshape(-1, 1)).reshape(-1) + 0.0000001;
        decision_curve = density * delta
        decision_curve_sort = np.argsort(decision_curve)[::-1]
        labels = np.full(len(self.distance_matrix), -1, dtype=int)
        for i in range(self.n_cluster):
            labels[decision_curve_sort[i]] = i
        for ele in indexListSortedByDensity:
            if labels[ele] == -1:
                labels[ele] = labels[neighborIndexList[ele]]
        return labels,decision_curve_sort[:self.n_cluster]

if __name__ == '__main__':
    X = np.array([[1, 2], [2, 2], [2, 3],
                  [8, 7], [8, 8], [25, 80]])
    result_list,centerIndex_list=DensityPeak(X=X,n_cluster=2).autoSearch()
    print(result_list)
    # print(centerIndex_list)
    #centerList=X[centerIndex_list]
    pass
