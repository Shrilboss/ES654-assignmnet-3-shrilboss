import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from Neural_network.network import NN
from metrics import *
from sklearn import linear_model
from sklearn.datasets import load_digits,load_boston
from sklearn.preprocessing import OneHotEncoder

np.random.seed(42)

#Use learning rate 0.6
#uncomment y_hat
X,y = load_boston(return_X_y=True)

from sklearn.preprocessing import MinMaxScaler         
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
y=y.reshape((len(y),1))
print(X.shape)
print(y.shape)
layers = [50,30]
activations = ["sigmoid","relu"]
t = NN(X,y,layers,activations)
t.train(X,y,3000)
y_hat = t.predict_regression(X)
print(y_hat.reshape(1,len(y)))
print(y_hat.shape)
print("MAE:",mae(y_hat,y))
print("RMSE:",rmse(y_hat,y))