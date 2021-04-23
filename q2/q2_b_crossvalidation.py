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

from sklearn.preprocessing import MinMaxScaler

#5-fold nested cross-validation
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

X,y = load_breast_cancer(return_X_y=True)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X.shape)
print(y.shape)

X_folds,y_folds = Generate_folds(X,y,5)
parameters = [0.005 , 0.05 , 0.5 , 5 , 50 , 500]
# parameters = [0.005 , 0.05 , 0.1 , 0.2 ,0.09]

test_fold = 0
optimum_lambda = []
for i in range(len(X_folds)):
    print("Fold :",str(i+1))
    testx = X_folds[test_fold]
    testy = y_folds[test_fold]
    trainx =[]
    trainy =[]
    for t in range(len(X_folds)):
        if(t!=test_fold):
            trainx.append(X_folds[t])
            trainy.append(y_folds[t])
    trainx=np.array(trainx)
    trainx=trainx.reshape((trainx.shape[0]*trainx.shape[1],trainx.shape[2]))
    trainy=np.array(trainy)
    trainy=trainy.reshape((trainy.shape[0]*trainy.shape[1],))
    # print(trainx.shape,trainy.shape)
    Avg_accuracy={}
    for lambda1 in parameters:
        subx_folds,suby_folds = Generate_folds(trainx,trainy,5)
        # print(subx_folds.shape,suby_folds.shape)
        sub_test_fold=0
        accuracy_list=[]
        for j in range(len(subx_folds)):
            print("SubFold :",str(j+1))
            sub_testx = subx_folds[sub_test_fold]
            sub_testy = suby_folds[sub_test_fold]
            sub_trainx =[]
            sub_trainy =[]
            for t in range(len(subx_folds)):
                if(t!=sub_test_fold):
                    sub_trainx.append(subx_folds[t])
                    sub_trainy.append(suby_folds[t])
            # print()
            sub_trainx=np.array(sub_trainx)
            sub_trainx=sub_trainx.reshape((sub_trainx.shape[0]*sub_trainx.shape[1],sub_trainx.shape[2]))
            sub_trainy=np.array(sub_trainy)
            sub_trainy=sub_trainy.reshape((sub_trainy.shape[0]*sub_trainy.shape[1],))
            # print(sub_trainx.shape,sub_trainy.shape)
            # print(sub_testx.shape,sub_testy.shape)
            LR = LogisticRegression(regularization="L1",C=lambda1)
            # LR = LogisticRegression(regularization="L2",C=lambda1)
            LR.fit_autograd(sub_trainx,sub_trainy)
            y_hat = LR.predict(sub_testx)
            curr_accuracy = accuracy(y_hat,sub_testy)
            accuracy_list.append(curr_accuracy)
            sub_test_fold+=1
        # print(accuracy_list)
        Avg_accuracy[lambda1]=np.mean(accuracy_list)
        # print("Lambda :",lambda1)
        # print(Avg_accuracy[lambda1])
    max_accuracy=-1
    opt_lambda=parameters[0]
    for o in Avg_accuracy.keys():
        if(Avg_accuracy[o]>max_accuracy):
            max_accuracy=Avg_accuracy[o]
            opt_lambda=o
    print("Opt lambda")
    print(opt_lambda)
    optimum_lambda.append(opt_lambda)
    test_fold+=1

print(optimum_lambda)