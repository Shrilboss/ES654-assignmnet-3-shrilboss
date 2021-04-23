#import modules
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.colors as cma
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
#import Autograd modules
import autograd.numpy as np
from autograd import grad,elementwise_grad

class LogisticRegression():
    def __init__(self,learning_rate=0.1, max_iter=1000, regularization='None', C = 0.8):
        """
        learning_rate, max_iter, regularization, C
        """
        self.alpha = learning_rate
        self.iterations = max_iter
        self.regularization = regularization
        self.C = C
        self.theta = None

    #For binary class problem
    def fit(self,X,y):
        """
        X.shape = (n_samples,n_features)
        y.shape = (n_samples,)
        """ 
        #define variables
        self.theta = np.zeros(X.shape[1]+1)
        X_new = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        n_features = X_new.shape[1]
        n_samples = X_new.shape[0]

        #Calculate grad for each iteration
        for i in range(self.iterations):
            X_t =np.dot(X_new,self.theta)
            y_hat = self.sigmoid(X_t)
            error = y_hat - y
            if(self.regularization == 'None'):
                grad = (self.alpha)*(np.dot(X_new.T,error))
            #update theta from the grad calculated
            self.theta -= grad/n_samples

    #When the classes are more than or equal to 2
    def fit_multiclass(self,X,y,K):
        """
        X.shape = (n_samples,n_features)
        y.shape = (n_samples,)
        """ 
        self.K = K
        X_new = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        one = OneHotEncoder(sparse=False)
        y_ = y.reshape(len(y),1)
        #One hot encode y
        y_new = one.fit_transform(y_)
        self.theta = np.zeros([X.shape[1]+1,y_new.shape[1]])
        n_samples = len(X_new)
        #Calculate softmax for each iteration
        for i in range(self.iterations):
            print(i)
            den = 0
            for k in range(y_new.shape[1]):
                den+= np.exp(np.dot(X_new,self.theta[:,k]))
            for j in range(y_new.shape[1]):
                softmax = np.exp(np.dot(X_new,self.theta[:,j]))/den
                error = softmax - y_new[:,j]
                #update theta with cross entropy loss
                self.theta[:,j] -= (self.alpha/n_samples)*(np.dot(error,X_new))

    #Defining the cost function
    def cost_function(self,X,y,theta):
        X_t =np.dot(X,theta)
        logistic = self.sigmoid(X_t)
        val = -1 * (y*(np.log(logistic)) + (1-y)*(np.log(1 - logistic)))
        return(np.mean(val))

    #Sigmoid function
    def sigmoid(self,z):
        return 1/(1+np.exp(-z)) 

    #fit funtion using autograd
    def fit_autograd(self,X,y):
        #Cost function for unregularised
        def cost(theta,X,y):
            X_t =np.dot(X,theta)
            logistic = self.sigmoid(X_t)
            val = -1 * (y*(np.log(logistic)) + (1-y)*(np.log(1 - logistic)))
            return(val.mean(axis=None))

        #Cost function for L2 regularisation
        def cost_2(theta,X,y):
            X_t =np.dot(X,theta)
            logistic = self.sigmoid(X_t)
            val = -1 * (y*(np.log(logistic)) + (1-y)*(np.log(1 - logistic)))
            g = np.sum(val)
            g += self.C*(np.dot(theta.T,theta))
            return(g/X.shape[0])

        #Cost function for L1 regularisation
        def cost_1(theta,X,y):
            X_t =np.dot(X,theta)
            logistic = self.sigmoid(X_t)
            val = -1 * (y*(np.log(logistic)) + (1-y)*(np.log(1 - logistic)))
            g = np.sum(val)
            g += self.C*(abs(theta))
            return(g/X.shape[0])

        #define differentiation functions
        grad_cost = grad(cost)
        grad_cost_1 = elementwise_grad(cost_1)
        grad_cost_2 = elementwise_grad(cost_2)

        self.theta = np.zeros(X.shape[1]+1)
        X_new = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        n_features = X_new.shape[1]
        n_samples = X_new.shape[0]

        #Find grad for each iteration
        for i in range(self.iterations):
            #update theta according to the regularisation provided
            if(self.regularization == 'None'):
                self.theta -= (self.alpha)*grad_cost(self.theta,X_new,y)
            elif(self.regularization == "L2"):
                self.theta -= (self.alpha)*grad_cost_2(self.theta,X_new,y) 
            elif(self.regularization == "L1"):
                self.theta -= (self.alpha)*grad_cost_1(self.theta,X_new,y) 

    #autograd multiclass function
    def fit_autograd_multiclass(self,X,y,K):
        #define crossentropy loss function
        def cost(theta,X,y):
            den = np.exp(np.dot(X,theta)).sum(axis=-1,keepdims=True)
            log_softmax = -(X@theta) + np.log(den)
            val = (y*log_softmax)
            return(val)
        self.K=K
        grad_cost =  elementwise_grad(cost)
        X_new = np.concatenate((np.zeros((X.shape[0], 1)), X), axis=1)
        one = OneHotEncoder(sparse=False)
        y_ = y.reshape(len(y),1)
        #one hot encode y
        y_new = one.fit_transform(y_)
        self.theta = np.zeros([X.shape[1]+1,y_new.shape[1]])
        n_features = X_new.shape[1]
        n_samples = X_new.shape[0]
        #update theta for each iteration
        for i in range(self.iterations):
            self.theta -= (self.alpha/n_samples)*grad_cost(self.theta,X_new,y_new)

    #prediction for binary class
    def predict(self,X):
        probability = self.sigmoid(np.dot(X,self.theta[1:])+self.theta[0])
        return (np.where(probability>=0.5,1,0))

    #prediction for multiple classes
    def predict_multiclass(self,X):
        probability = self.sigmoid(np.dot(X,self.theta[1:])+self.theta[0])
        vals = np.argmax(probability,axis=1)
        return vals

    #plot 2d decision boundaru
    def plot_decision_boundary(self,X, y):
        cMap = cma.ListedColormap(["#6666ff", "#ff8080"])

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = .02  # step size in the mesh

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = self.predict(np.column_stack((xx.ravel(), yy.ravel())))
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(8, 6), frameon=True)
        plt.axis('off')
        plt.pcolormesh(xx, yy, Z, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=y,s=30, marker = "o", edgecolors='k', cmap=cMap)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()