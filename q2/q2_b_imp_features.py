import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from LogisticRegression.logisticregression import LogisticRegression
from metrics import *
from sklearn import linear_model
np.random.seed(5)
from sklearn.datasets import load_breast_cancer,load_digits,load_boston,make_classification
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler


X,y = load_breast_cancer(return_X_y=True)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
LR = LogisticRegression(regularization="L1")
LR.fit_autograd(X,y)
tolerance = 10
imp_features = []
for i in range(1,X.shape[1]):
    val = (LR.theta[i])**2
    # print(val,i)
    if(val > tolerance):
        imp_features.append(i)
print("IMP FEATURES")
print(imp_features)

X,y = load_breast_cancer(return_X_y=True)
max_accuracy = -1
best_feature = 0
accuracy_list =[]
for features in range(X.shape[1]):
    X,y = load_breast_cancer(return_X_y=True)
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X = X[:,features]
    X = X.reshape((len(X),1))
    LR = LogisticRegression(regularization="L1")
    LR.fit_autograd(X,y)
    y_hat = LR.predict(X)
    curr_accuracy = accuracy(y_hat,y)
    print(curr_accuracy)
    accuracy_list.append((curr_accuracy,features+1))
    if(curr_accuracy>max_accuracy):
        max_accuracy=curr_accuracy
        best_feature=features
print(max_accuracy," For feature ",best_feature+1)

accuracy_list.sort(reverse=True)
print("BEST 5 features:")
print([x[1] for x in accuracy_list[:5]])


