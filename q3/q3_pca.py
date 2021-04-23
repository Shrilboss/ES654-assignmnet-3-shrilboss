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
# X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,n_clusters_per_class=1)
# X,y = load_breast_cancer(return_X_y=True)
X,y = load_digits(return_X_y=True)

from sklearn.preprocessing import MinMaxScaler     

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pca.fit_transform(X)

def plot_pca():
    colors = ['yellow','red','blue','purple','gold','black','pink','green','cyan','orange']
    for i in range(len(colors)):
        px = X[:,0][y == i]
        py = X[:,1][y == i]
        plt.scatter(px,py,c=colors[i],s=30,edgecolors="black",linewidths=1)
    plt.legend([str(r) for r in range(10)])
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
plt.title("PCA on Digits dataset (n_components =2)")
plot_pca()
# plt.savefig("D:\imp docs\study\Semesters\sem 6\ML\Assignments\ES654-assignmnet-3-shrilboss\pca_scatterplot.jpg")
plt.show()