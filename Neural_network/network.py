#import modules
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.colors as cma
from scipy.special import softmax,log_softmax
from sklearn.metrics import mean_squared_error
from scipy.special import expit
from Neural_network.layers import *
#import Autograd modules
import autograd.numpy as np
from autograd import grad,elementwise_grad
np.random.seed(3)

#Define the neural network
class NN():

    def intialise_params(self,layer_dims):
        #initialise weights and bias for all layers
        parameters = {}
        layers = layer_dims
        total_layers = len(layers) # number of layers in the network

        # print("-------------Params---------")
        for i in range(1, total_layers):
            parameters['W' + str(i)]=np.random.normal(loc=0.0,scale = np.sqrt(2/(layers[i-1]+layers[i])),size = (layers[i-1],layers[i]))
            parameters['b' + str(i)] = np.zeros((1,layers[i]))     
        return parameters
    
    def __init__(self,inputs,output,layer_dims,activations,learning_rate=4):
        """
        input,output,layer_dims,activations
        """
        self.X = inputs
        self.y = output
        self.learning_rate = learning_rate
        self.layer_dims = layer_dims
        self.activations = activations
        full_dims = [inputs.shape[1]]
        full_dims.extend(layer_dims)
        full_dims.append(output.shape[1])
        self.parameters = self.intialise_params(full_dims)

    def forward(self,input):
        #apply forward pass to neural network and return the output list
        self.caches = []
        A_current = input
        # print("-------------Activations---------")
        activations_list = []
        L = len(self.parameters)//2
        #iterate over each layer and calculate its output
        for i in range(1,L):
            A_previous = A_current
            A_current,cache = forward_activation(A_previous,self.parameters["W"+str(i)],self.parameters["b"+str(i)],activation=self.activations[i-1])
            self.caches.append(cache)
            activations_list.append(A_current)
        output,cache = forward_activation(A_current,self.parameters["W"+str(L)],self.parameters["b"+str(L)],activation="identity")
        activations_list.append(output)
        self.caches.append(cache)

        #for K_class classification
        s = self.softmax(output)
        activations_list.append(s)

        #for Binary classfication
        # s = sigmoid(output)[0]
        # activations_list.append(s)

        return(activations_list)

    def backward(self,grad_loss):
        #backward pass the neural netwwork and update parameters
        L = len(self.parameters)//2
        for i in reversed(range(L-1)):
            curr_cache = self.caches[i]
            curr_activation = self.activations[i]
            #calculate grad_loss frm previous grad
            grad_loss , dW , db = backward_activation(grad_loss,curr_cache,activation =curr_activation)
            
            #update params
            self.parameters["W"+str(i+1)]-= self.learning_rate*dW
            self.parameters["b"+str(i+1)]-= self.learning_rate*db

    def predict(self,input):
        #predict the output from input for binary classification
        activation_list = self.forward(input)
        probability = activation_list[-1]
        return(np.where(probability>=0.5,1,0))
    
    def predict_regression(self,input):
        #predict output for regression
        activation_list = self.forward(input)
        probability = activation_list[-1]
        return(probability)

    def predict_multiclass(self,input):
        #predict output for multiclass problems
        activations = self.forward(input)
        probability = activations[-1]
        vals = np.argmax(probability,axis=1)
        return vals

    def softmax(self,y):
        #inbuilt scipy softmax function
        return(softmax(y))

    def log_softmax(self,y):
        #inbuilt scipy log_softmax function
        return(log_softmax(y))

    def train(self,X,y,epochs):
        #training the neural network for given number of epochs
        #define cost functions for autograd
        def softmax(output,y):
            print(output.shape)
            log_softmax = -output + np.log(np.sum(np.exp(output),axis=-1))
            val = (y*log_softmax)
            return val

        def cross_entropy(output,y):
            r = np.log(output)
            val = (y*r)
            return(-val)

        def mse_prime(y_true, y_pred):
            return 2*(y_pred-y_true)/y_true.size;
        def mse(output,y):
            return np.mean(np.power(y-output,2))
            # return(mean_squared_error(y,output))

        #run the network for each epoch
        for epoch in range(epochs):

            print("Epoch :",epoch,end="\r")
            activation_list = self.forward(X)

            #for Digits
            y_hat = activation_list[-2]

            #for Boston
            # y_hat = activation_list[-1]

            #Calculate grad_loss for last layer
            grad_cost =  elementwise_grad(mse)
            grad_loss = grad_cost(y_hat,y)

            curr_cache = self.caches[-1]
            A_prev , Weights, bias = curr_cache
            n_features = A_prev.shape[1] 
            # print(grad_loss.shape , A_prev.shape , Weights.shape)
            #Calculate dW, db and dA_prev
            dW = (1/n_features)*np.dot(A_prev.T,grad_loss)
            db = (1/n_features)*np.sum(grad_loss,axis=0,keepdims=True)
            dA_prev = np.dot(grad_loss,Weights.T)
            
            L = len(self.parameters)//2
            # update params
            self.parameters["W"+str(L)]-= self.learning_rate*dW
            self.parameters["b"+str(L)]-= self.learning_rate*db

            #run backward and update params
            self.backward(dA_prev)