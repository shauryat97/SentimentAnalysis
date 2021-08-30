import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import re
from preprocess import preprocess,listToString
from split import split_set
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Split datasets into training and test

review_text = open(r'/Users/shauryatiwari/Desktop/work/ass1_2/movieReviews1000.txt','r')
train_set,test_set = split_set(review_text)

# Preprocessing - remove special characters , remove stopwords ,  lemmatize , tokenize.

train_docs , train_labels = preprocess(train_set)
test_docs , test_labels = preprocess(test_set)
s= 0 
for i in range(len(train_docs)):
    s+=len(train_docs[i])
print(s)
train_lst = []
for i in range(len(train_docs)):
    string = listToString(train_docs[i])
    train_lst.append(string)
test_lst = []
for i in range(len(test_docs)):
    string = listToString(test_docs[i])
    test_lst.append(string)

# Use tf-idf to extract features
# Python program to convert a list to string
    
# Function to convert  


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_lst)
test_vectors = vectorizer.transform(test_lst)
train_dense = train_vectors.todense()
test_dense = test_vectors.todense()
print(train_dense.shape)
# PCA to reduce dimension

# Therefore I would suggest (analogously to the common mean imputation of missing values) to perform TF-IDF-normalization on the training set seperately
#  and then use the IDF-vector from the training set to calculate the TF-IDF vectors of the test set.


# Train SVM model and test on test set - try different kernels - rbf,poly,linear

