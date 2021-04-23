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
X,y = load_breast_cancer(return_X_y=True)
# X,y = load_digits(return_X_y=True)

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


# LR = LogisticRegression(regularization="None")
# LR.fit_multiclass(X,y,n_class)
# # LR.fit_autograd_multiclass(X,y,n_class)
# y_hat = LR.predict_multiclass(X)
# print(y_hat)
# ac = accuracy(y_hat,y)
# print(ac)
# cm = confusion_matrix(y,y_hat)
# print(confusion_matrix(y,y_hat))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)
# disp.plot()
# # plt.savefig("confusionmatrix_3c.jpg")
# plt.show()

# LR.plot_decision_boundary(X, y)
# reg = linear_model.LogisticRegression()
# reg.fit(X,y)
# y_hat = reg.predict(X)
# print(accuracy(y_hat,y))



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

max_accuracy = -1
best_feature = 0
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
    if(curr_accuracy>max_accuracy):
        max_accuracy=curr_accuracy
        best_feature=features
print(max_accuracy," For feature ",best_feature+1)

# X_folds,y_folds = Generate_folds(X,y,5)
# parameters = [0.005 , 0.05 , 0.5 , 5 , 50 , 500]
# # parameters = [0.005 , 0.05 , 0.1 , 0.2 ,0.09]

# test_fold = 0
# optimum_lambda = []
# for i in range(len(X_folds)):
#     testx = X_folds[test_fold]
#     testy = y_folds[test_fold]
#     trainx =[]
#     trainy =[]
#     for t in range(len(X_folds)):
#         if(t!=test_fold):
#             trainx.append(X_folds[t])
#             trainy.append(y_folds[t])
#     trainx=np.array(trainx)
#     trainx=trainx.reshape((trainx.shape[0]*trainx.shape[1],trainx.shape[2]))
#     trainy=np.array(trainy)
#     trainy=trainy.reshape((trainy.shape[0]*trainy.shape[1],))
#     # print(trainx.shape,trainy.shape)
#     Avg_accuracy={}
#     for lambda1 in parameters:
#         subx_folds,suby_folds = Generate_folds(trainx,trainy,5)
#         # print(subx_folds.shape,suby_folds.shape)
#         sub_test_fold=0
#         accuracy_list=[]
#         for j in range(len(subx_folds)):
#             sub_testx = subx_folds[sub_test_fold]
#             sub_testy = suby_folds[sub_test_fold]
#             sub_trainx =[]
#             sub_trainy =[]
#             for t in range(len(subx_folds)):
#                 if(t!=sub_test_fold):
#                     sub_trainx.append(subx_folds[t])
#                     sub_trainy.append(suby_folds[t])
#             # print()
#             sub_trainx=np.array(sub_trainx)
#             sub_trainx=sub_trainx.reshape((sub_trainx.shape[0]*sub_trainx.shape[1],sub_trainx.shape[2]))
#             sub_trainy=np.array(sub_trainy)
#             sub_trainy=sub_trainy.reshape((sub_trainy.shape[0]*sub_trainy.shape[1],))
#             # print(sub_trainx.shape,sub_trainy.shape)
#             # print(sub_testx.shape,sub_testy.shape)
#             # LR = LogisticRegression(regularization="L1",C=lambda1)
#             LR = LogisticRegression(regularization="L2",C=lambda1)
#             LR.fit_autograd(sub_trainx,sub_trainy)
#             y_hat = LR.predict(sub_testx)
#             curr_accuracy = accuracy(y_hat,sub_testy)
#             accuracy_list.append(curr_accuracy)
#             sub_test_fold+=1
#         # print(accuracy_list)
#         Avg_accuracy[lambda1]=np.mean(accuracy_list)
#         print("Lambda :",lambda1)
#         print(Avg_accuracy[lambda1])
#     max_accuracy=-1
#     opt_lambda=parameters[0]
#     for o in Avg_accuracy.keys():
#         if(Avg_accuracy[o]>max_accuracy):
#             max_accuracy=Avg_accuracy[o]
#             opt_lambda=o
#     print("Opt lambda")
#     print(opt_lambda)
#     optimum_lambda.append(opt_lambda)
#     test_fold+=1

# print(optimum_lambda)