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
from pca_utils import *
from svm_utils import *
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os


cwd = os.getcwd()
cwd+='/movieReviews1000.txt' # don't change file name


# Split datasets into training and test

review_text = open(cwd,'r')
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
train_dense = train_vectors.todense() #convert from scipy matrix to numpy matrix
test_dense = test_vectors.todense()


# PCA to reduce dimension
k = 10 # Number of components.
N_train = train_dense.shape[0]
train_dense,mean = get_XafterMean(train_dense,N_train)
test_dense= get_XafterMean_test(test_dense,mean)
train_dense,test_dense = calc_pca(train_dense,test_dense,k)


# Train SVM model and test on test set - try different kernels - rbf,poly,linear

train_labels = np.array(train_labels).reshape(N_train,-1)
test_labels = np.array(test_labels).reshape(len_test,-1)

def main_svm(train_dense,test_dense,train_labels,test_labels,lst_kernel):

    for kernel in lst_kernel:
        y_pred,n_support = svm_model(train_dense,train_labels,test_dense,kernel)
        y_pred = y_pred.reshape(y_pred.shape[0],-1)
        out_string = ' '+kernel+' '+'kernel '
        print(out_string+'\n')
        acc = np.sum((np.equal(y_pred,test_labels)))
        acc= acc/y_pred.shape[0]
        print('acc',round(acc*100,4),'\n')
        n_support = sum(n_support)
        print(' Number of Support Vectors =  ',n_support,'\n')

lst_kernel = ['linear','rbf','poly']
main_svm(train_dense,test_dense,train_labels,test_labels,lst_kernel)

