import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import re
from preprocess import preprocess,listToString,string_lst
from split import split_set
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from pca import *
from svm import *
import warnings
warnings.filterwarnings('ignore')

# Split datasets into training and test

review_text = open(r'/Users/shauryatiwari/Desktop/work/ass1_2/movieReviews1000.txt','r')
train_set,test_set,len_test = split_set(review_text)

# Preprocessing - remove special characters , remove stopwords ,  lemmatize , tokenize.

train_docs , train_labels = preprocess(train_set)
test_docs , test_labels = preprocess(test_set)

train_lst = string_lst(train_docs)
test_lst = string_lst(test_docs)


# Use tf-idf to extract features

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_lst)
test_vectors = vectorizer.transform(test_lst)
train_dense = train_vectors.todense() #numpy matrix
test_dense = test_vectors.todense()

# print(len(vectorizer.get_feature_names()))
# print(train_dense.shape)

# PCA to reduce dimension

N_train = train_dense.shape[0]
XafterMean_train,mean = get_XafterMean(train_dense,N_train)
XX_T = get_XXtranspose(XafterMean_train)
eigenval,eigenvec = eigen(XX_T/N_train)
eigenval,eigenvec  = eigsorted(eigenval,eigenvec)
U = get_U(N_train,XafterMean_train,eigenvec)
Uprime = U[:,:10] # Taking first k eigen vectors corresponding to k largest eigen values ; Here k=10
Y_train,pca = calc_pca(XafterMean_train)
# Y_train = toPCA(N_train,XafterMean_train,Uprime,10)


# For test data
XafterMean_test = get_XafterMean_test(test_dense,mean)
Uprime_test = U[:,:10]
Y_test = pca.transform(XafterMean_test)
# Y_test = toPCA(len_test,XafterMean_test,Uprime_test,10)
# # Therefore I would suggest (analogously to the common mean imputation of missing values) to perform TF-IDF-normalization on the training set seperately
# #  and then use the IDF-vector from the training set to calculate the TF-IDF vectors of the test set.


# # Train SVM model and test on test set - try different kernels - rbf,poly,linear
train_labels = np.array(train_labels).reshape(N_train,-1)
test_labels = np.array(test_labels).reshape(len_test,-1)

def main(lst_kernel):
    for kernel in lst_kernel:
        y_pred,n_support = svm_model(Y_train,train_labels,Y_test,kernel)
        y_pred = y_pred.reshape(y_pred.shape[0],-1)
        target_names = ['class 0', 'class 1']
        out_string = ' '+kernel+' '+'kernel '
        print(out_string+'\n\n')
        acc = np.sum(np.abs(y_pred-test_labels))
        print(1-(acc/y_pred.shape[0]))
        n_support = sum(n_support)
        print(' Number of Support Vectors =  ',n_support)

lst_kernel = ['linear','rbf','poly']
# main(lst_kernel)



# seperate 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_dense)
# Apply transform to both the training set and the test set.
train_dense= scaler.transform(train_dense)
test_dense = scaler.transform(test_dense)
print('before',train_dense.shape)

pca = PCA(n_components=10)
pca.fit(train_dense)

train_dense = pca.transform(train_dense)
test_dense = pca.transform(test_dense)
print('after',train_dense.shape)
main(lst_kernel)

