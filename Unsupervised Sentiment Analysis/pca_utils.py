import numpy as np
from sklearn.decomposition import PCA 

def get_XafterMean(X,N):  
    mean = np.sum(X,axis= 0)/N
    mean = np.ravel(mean)
    XafterMean = X.copy()
    for j in range(X.shape[1]):
        for i in range(X.shape[0]):
            XafterMean[i,j]=X[i,j] - mean[j]
    return XafterMean,mean

def calc_pca(train,test,k):
    
    pca = PCA(n_components = k)
    pca.fit(train)
    train = pca.transform(train)
    test = pca.transform(test)
    return train,test