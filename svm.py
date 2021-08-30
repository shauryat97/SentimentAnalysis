from sklearn.metrics import classification_report
from sklearn.svm import SVC
 
def svm_model(x_train,y_labels,x_test,kernel):
    clf = SVC(
    kernel = kernel)
    clf.fit(x_train,y_labels)
    y_pred = clf.predict(x_test)
    return y_pred,clf.n_support_

