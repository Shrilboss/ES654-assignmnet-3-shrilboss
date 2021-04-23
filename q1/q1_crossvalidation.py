#import modules
import numpy as np
import pandas as pd
import sys
sys.path.append(".")
import matplotlib.pyplot as plt
from LogisticRegression.logisticregression import LogisticRegression
from metrics import *
from sklearn import linear_model
np.random.seed(5)
from sklearn.datasets import load_breast_cancer,load_digits,load_boston,make_classification
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
X,y = load_breast_cancer(return_X_y=True)

#generate n folds from x and y
def Generate_folds(X,y,folds):
    X_folds=[]
    y_folds=[]
    k=0
    for i in range(folds):
        xlist=[]
        ylist=[]
        size = int(len(X)/folds)
        for j in range(k,k+size):
            xlist.append(X[j])
            ylist.append(y[j])
        X_folds.append(xlist)
        y_folds.append(ylist)
        k+=size
    X_folds=np.array(X_folds)
    y_folds=np.array(y_folds)
    return(X_folds,y_folds)

from sklearn.preprocessing import MinMaxScaler     
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
n_class = len(np.unique(y)) 
classes = np.unique(y)
print(X.shape)
print(y.shape)
x_folds , y_folds = Generate_folds(X,y,3)
test_fold =0
accuracy_list =[]
print("-------Cross Validation-------")
for i in range(len(x_folds)):
    print("Fold :",str(i+1))
    testx = x_folds[test_fold]
    testy = y_folds[test_fold]
    trainx =[]
    trainy =[]
    for t in range(len(x_folds)):
        if(t!=test_fold):
            trainx.append(x_folds[t])
            trainy.append(y_folds[t])
    trainx=np.array(trainx)
    trainx=trainx.reshape((trainx.shape[0]*trainx.shape[1],trainx.shape[2]))
    trainy=np.array(trainy)
    trainy=trainy.reshape((trainy.shape[0]*trainy.shape[1],))
    LR = LogisticRegression(regularization="None")
    LR.fit_autograd(trainx,trainy)
    y_hat = LR.predict(testx)
    curr_accuracy = accuracy(y_hat,testy)
    print("Accuracy :",curr_accuracy)
    accuracy_list.append(curr_accuracy)
    test_fold+=1

print("")
print("Overall Accuracy:")
print("Max Accuracy :",max(accuracy_list))
print("Average Accuracy :",np.mean(accuracy_list))