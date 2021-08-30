import numpy as np



def get_XafterMean(X,N):  
    mean = np.sum(X,axis= 0)/N
    mean = np.ravel(mean)
    XafterMean = X.copy()
    for j in range(X.shape[1]):
        for i in range(X.shape[0]):
            XafterMean[i,j]=X[i,j] - mean[j]
    return XafterMean,mean

def get_XXtranspose(XafterMean):
    return XafterMean@np.transpose(XafterMean)



def eigen(XX_t):
    return  np.linalg.eig(XX_t)

def eigsorted(eigenval,eigenvec):
    indexInDescendingOrder = eigenval.argsort()[::-1]   
    eigenval = eigenval[indexInDescendingOrder]
    eigenvec = eigenvec[:,indexInDescendingOrder]
    return eigenval,eigenvec


def get_U(N,XafterMean,eigenvec):
    U =np.zeros((10201,N))
    for i in range(N):
        U[:,i]=(XafterMean.T)@eigenvec[:,i]
        U[:,i]/= np.linalg.norm(U[:,i],ord = 2)
    return U

#Compute PCA
def toPCA(N,XafterMean,Uprime,k):
    Y = []
    for i in range(N):
        yi = Uprime.T.dot(XafterMean[i,:])
        yi.tolist()
        Y.append(yi)
    Y = np.array(Y)
    Y = np.reshape(Y,(N,k))
    return Y