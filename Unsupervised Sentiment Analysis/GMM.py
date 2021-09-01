import numpy as np
from scipy.stats import multivariate_normal
from pca_utils import *
import nltk
from sklearn.cluster import KMeans
from preprocess import preprocess,listToString,string_lst
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')
import os
cwd = os.getcwd()
cwd+='/movieReviews1000.txt' # don't change file name

# Split datasets into training and test

review_text = open(cwd,'r')
lines_lst = review_text.readlines() # will return a list of reviews.
text_set = []
for i in range(1000):
    text_set.append(lines_lst[i])


# Preprocessing - remove special characters , remove stopwords ,  lemmatize , tokenize.

text_docs , text_labels = preprocess(text_set)
text_lst = string_lst(text_docs)

# Use tf-idf to extract features

vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(text_lst)
text_dense = text_vectors.todense() #convert from scipy matrix to numpy matrix

# PCA to reduce dimension

k = 10 # Number of components.
N_train = text_dense.shape[0]
text_dense,mean = get_XafterMean(text_dense,N_train)
pca = PCA(n_components=k)
text_dense = pca.fit_transform(text_dense)

# GMM

def create_sigma_alpha(m1,m2,labels,fm_afterPCA):
    class_1Index = np.where(labels ==1)[0] # returns a list of index of class 1
    class_2Index = np.where(labels==0)[0] #returns a list of index of class 2 i.e class 0
    class_1 = fm_afterPCA[class_1Index]
    class_2 = fm_afterPCA[class_2Index]
    sigma1 = np.cov(class_1.T)
    sigma2= np.cov(class_2.T)
    num_c1 = class_1.shape[0]
    num_c2 = class_2.shape[0]
    alpha1 = num_c1/1000
    alpha2 = num_c2/1000
    assert (alpha1+alpha2 ==1)
    return sigma1,sigma2,alpha1,alpha2

def calc_kmeans(fm_afterPCA):
    kmeans = KMeans(n_clusters=2,random_state=0).fit(fm_afterPCA)
    labels=kmeans.labels_
    mu=kmeans.cluster_centers_
    mu1 = mu[0,:]
    mu2 = mu[1,:]
    mu_lst=[mu1,mu2]
    sigma1,sigma2,alpha1,alpha2 = create_sigma_alpha(mu1,mu2,labels,fm_afterPCA)
    return sigma1,sigma2,alpha1,alpha2,mu1,mu2


class GMM():

    def __init__(self,k,n_iterations,tol,x):
        self.k = k
        self.n_iterations = n_iterations
        self.tol = tol
        self.initial_params(x)
    def calc_log_likelihood(self,x):
        for j in range(self.k):
            lambda_jj = self.lambda_j[j]
            pdf = multivariate_normal(self.means[j], self.covs[j],allow_singular=True).pdf(x)
            self.gamma_ij[:,j] = lambda_jj *pdf 
        return self    
    def e_step(self,x):
        self.calc_log_likelihood(x)
        log_likelihood = np.sum(np.log(np.sum(self.gamma_ij,axis=1))) # 𝑙𝑛(𝑝(𝑋∣𝜋,𝜇,Σ))=∑𝑖=1𝑁𝑙𝑛{∑𝑗=1𝐾𝜋𝑗𝑁(𝑥𝑖∣𝜇𝑗,Σ𝑗)}
        den = self.gamma_ij.sum(axis = 1,keepdims = 1)
        self.gamma_ij = self.gamma_ij/den # calculaed gamma_ij based on ith iteration.
        return log_likelihood
    def m_step(self,x):
        self.lambda_j = self.gamma_ij.sum(axis=0)/self.n # update values of lambda_j
        self.means = np.dot(self.gamma_ij.T,x)/self.gamma_ij.sum(axis=0).reshape(-1,1) # updated value for means
        for w in range(self.k):
            self.covs[w] = np.dot(self.gamma_ij[:,w] * (x-self.means[w]).T, (x-self.means[w]))
        return self 

    def initial_params(self,x):
        self.n,self.m = x.shape
        np.random.seed(0)
        chosen = np.random.choice(self.n, self.k, replace = False)
        self.means = x[chosen] # selecting any two points as mean 
        self.lambda_j = np.full(self.k,1 / self.k) # intitally giving 1/k weights to both the classes
        self.covs = np.full((self.k,self.m,self.m), np.cov(x, rowvar = False)) # sample covariance
    def fit(self,x):
        self.gamma_ij = np.zeros((self.n,self.k))
        log_likelihood = 0
        for i in range(self.n_iterations):
            this_step = self.e_step(x)
            self.m_step(x)
            if abs(this_step - log_likelihood) <= self.tol :
                break
            log_likelihood = this_step
        return self

def calc_accuracy(likelihood1,likelihood2,text_dense,out_str):
    lst_y_hat = []
    for i in range(text_dense.shape[0]):
        if likelihood1[i]>likelihood2[i]:
            lst_y_hat.append(0)
        else : 
            lst_y_hat.append(1)
    lst_y_hat = np.array(lst_y_hat)
    acc = np.sum(np.equal(lst_y_hat,text_labels))
    out_str = ' Accuracy for '+out_str+' Initialization = '
    print(out_str, acc/1000)

gmm = GMM(2,5,1e-3,text_dense)
gmm.fit(text_dense)
lambda_j = gmm.lambda_j
means = gmm.means
covs = gmm.covs
try:
    likelihood1 = multivariate_normal(means[0],covs[0],allow_singular=True).pdf(text_dense)
    likelihood2 = multivariate_normal(means[1],covs[1],allow_singular=True).pdf(text_dense)
    calc_accuracy(likelihood1,likelihood2,text_dense,'Random')
except:
    print('Nan Value , run again ')


# Using Library Fucntion ; KMeans Initialization also included.
from sklearn.mixture import GaussianMixture
gmm_library = GaussianMixture(n_components=2,tol=1e-3,max_iter=10,init_params='kmeans',covariance_type='diag', random_state=0).fit(text_dense)
y_hat = gmm_library.predict(text_dense)
acc_km = np.sum(np.equal(y_hat,text_labels))
print('Accuracy for KMeans Initialization =',acc_km/1000)