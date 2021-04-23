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

x_folds , y_folds = Generate_folds(X,y,3)
test_fold =0
rmse_list =[]
mae_list =[]
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
    trainy=trainy.reshape((len(trainy),1))
    t = NN(trainx,trainy,layers,activations)
    t.train(trainx,trainy,3000)
    y_hat = t.predict_regression(testx)
    curr_rmse = rmse(y_hat,testy)
    curr_mae = mae(y_hat,testy)
    print("MAE :",curr_mae)
    print("RMSE :",curr_rmse)
    rmse_list.append(curr_rmse)
    mae_list.append(curr_mae)
    test_fold+=1

print("")
print("Overall Accuracy:")
print("MIN MAE :",min(mae_list))
print("Average MAE :",np.mean(mae_list))
print("------------")
print("MIN RMSE:",min(rmse_list))
print("Average RMSE :",np.mean(rmse_list))