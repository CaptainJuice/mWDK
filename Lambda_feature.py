import numpy as np
from Lambda_map import *
from sklearn.preprocessing import normalize as nor

"""
def normalization_vect(vect,psi,t):
    for i in range(t):
        vect[:,int(i*psi):int((i+1)*psi)] = nor(vect[:,int(i*psi):int((i+1)*psi)])
    vect = vect/np.sqrt(t)
    return vect
"""

def normalization_row(vect,eta,psi,t):
    for i in range(t):
        tmp = vect[int(i*psi):int((i+1)*psi)]
        tmp_m = np.expand_dims(tmp,0).repeat(psi,axis=0)
        m = tmp_m-np.reshape(tmp,(-1,1))
        m = 1/np.sqrt(np.sum(np.exp(-2*eta*m),axis=1))
        assert np.all(m<=1)
        assert np.all(m>=0)
        vect[int(i*psi):int((i+1)*psi)] = m
    return vect/np.sqrt(t)

def normalization_vect(vect,eta,psi,t):
    assert vect.shape[1] == (psi*t)
    return np.array([normalization_row(x,eta,psi,t) for x in vect])

def lambda_feature_infty(distribution,newdata,psi,t=100):
    # produce feature and distance matrix for X and query_points
    lm = Lambda_map(psi,t)
    subIndex = lm.fit(distribution)
    dis_map = lm.transform(distribution).toarray()
    new_map = lm.transform(newdata).toarray()
    dis_map = dis_map/np.sqrt(t)
    new_map = new_map/np.sqrt(t)
    dm = 1-np.dot(dis_map,dis_map.T)
    dm[np.where(dm<0)]=0
    dm2 = 1-np.dot(new_map,dis_map.T)
    dm2[np.where(dm2<0)]=0
    return dis_map,new_map,dm,dm2,subIndex

def lambda_feature_continous(distribution,newdata,eta,psi,t=100):
    # produce feature and distance matrix for X and query_points
    lm = Lambda_map(psi,t)
    subIndex = lm.fit(distribution)
    dis_map = lm.transform_continous(distribution,eta).toarray()
    dis_map = normalization_vect(dis_map,eta,psi,t)
    new_map = lm.transform_continous(newdata,eta).toarray()
    new_map = normalization_vect(new_map,eta,psi,t)
    dm = 1-np.dot(dis_map,dis_map.T)
    dm[np.where(dm<0)]=0
    dm2 = 1-np.dot(new_map,dis_map.T)
    dm2[np.where(dm2<0)]=0
    return dis_map,new_map,dm,dm2,subIndex
