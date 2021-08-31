# Supervised and Unsupervised Sentiment Analysis

# Task
Given a corpus of movie reviews, train a model to predict sentimemt of the review.

# Dataset
Movie Reviews Dataset [data_link](http://www.leap.ee.iisc.ac.in/sriram/teaching/MLSP21/assignments/movieReviews1000.txt)
Each Line is a review with a label(0 or 1) associated at the end of the review.

# Supervised Approach

# Steps involved
-**Split the data into two subsets. One for training (first 750 reviews) and the other for testing (last 250 reviews). 
- **Use TF-IDF to vectorize your input.
- **Reduce dimension of feature vector to 10 by appying Principal Component Analysis(PCA).
- **Train a SVM model. And check the performance on the test set in terms of review classification accuracy.
- **Compare different kernel choices - linear, polynomial and radial basis function.
- **Report the number of support vectors used and the classification performance for different kernel choices.

# Tools
- Python Version: 3.8.8
- Numpy Version: 1.20.0
- Sklearn Version: 0.24.1
- NLTK Version: 3.6.1

# Results

| Kernel       | Classification Accuracy           | Number Of Support Vectors |
| ------------- |:-------------:| -----:|
| Linear      | 58.33 | 623 |
| Rbf  | **65.33**      |   **597** |
| Poly   | 56     | 596  |
