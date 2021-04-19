import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LogisticRegression.logisticregression import LogisticRegression
from metrics import *
from sklearn import linear_model
np.random.seed(5)
from sklearn.datasets import load_breast_cancer,load_digits,load_boston,make_classification
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
# X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,n_clusters_per_class=1)
# X,y = load_breast_cancer(return_X_y=True)
X,y = load_digits(return_X_y=True)

from sklearn.preprocessing import MinMaxScaler         
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X.shape)
print(y.shape)

# print(X[:,:2].shape)
# X,y = load_breast_cancer(return_X_y=True)
# X = X[:,12:14]
n_class = len(np.unique(y)) 
classes = np.unique(y)
LR = LogisticRegression(regularization="None")
# LR.fit_multiclass(X,y,n_class)
LR.fit_autograd_multiclass(X,y,n_class)
y_hat = LR.predict_multiclass(X)
print(y_hat)
ac = accuracy(y_hat,y)
print(ac)
cm = confusion_matrix(y,y_hat)
print(confusion_matrix(y,y_hat))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)
disp.plot()
# plt.savefig("confusionmatrix_3c.jpg")
plt.show()

# LR.plot_decision_boundary(X, y)
# reg = linear_model.LogisticRegression()
# reg.fit(X,y)
# y_hat = reg.predict(X)
# print(accuracy(y_hat,y))