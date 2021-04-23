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

from sklearn.preprocessing import MinMaxScaler     
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
n_class = len(np.unique(y)) 
classes = np.unique(y)

print(X.shape)
print(y.shape)
#Enter the features to plot
feature1 = 7
feature2 = 17
X = np.hstack((X[:,feature1].reshape((len(X),1)),X[:,feature2].reshape((len(X),1))))
print(X.shape)
LR = LogisticRegression(regularization="None")
LR.fit(X,y)
y_hat = LR.predict(X)
LR.plot_decision_boundary(X, y)