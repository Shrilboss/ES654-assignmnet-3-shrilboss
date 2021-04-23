import numpy as np
import pandas as pd
import sys
sys.path.append(".")
import matplotlib.pyplot as plt
from Neural_network.network import NN
from metrics import *
from sklearn import linear_model
from sklearn.datasets import load_digits,load_boston
from sklearn.preprocessing import OneHotEncoder

np.random.seed(42)
#uncomment K-class classification
#Use learning rate 20
#uncomment y_hat
X,y = load_digits(return_X_y=True)

from sklearn.preprocessing import MinMaxScaler         
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
one = OneHotEncoder(sparse=False)
y_ = y.reshape(len(y),1)
y_new = one.fit_transform(y_)
print(X.shape)
print(y_new.shape)
layers = [64,32]
activations = ["sigmoid","relu"]
t = NN(X,y_new,layers,activations)
t.train(X,y_new,2000)
y_hat = t.predict_multiclass(X)
print(y_hat)
print(y_hat.shape)
print(accuracy(y_hat,y))
