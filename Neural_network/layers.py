import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.colors as cma

np.random.seed(3)

def sigmoid(Z):
    """
    Z -- numpy array of any shape
    returns
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    """
    Z -- Output of the linear layer, of any shape
    returns
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    A = np.maximum(0,Z)
    cache = Z 
    return A, cache

def back_relu(prev_grad,cache):
    """
    prev_grad - gradient of previous layer
    cache -- stored for computing the backward pass efficiently
    Output - relu gradient
    """
    Z = cache
    dZ = np.array(prev_grad,copy=True)
    dZ[Z<=0] = 0
    return(dZ)

def back_sigmoid(prev_grad,cache):
    """
    prev_grad - gradient of previous layer
    cache -- stored for computing the backward pass efficiently
    Output - sigmoid gradient
    """
    Z = cache
    sigmoid = 1/(1+np.exp(-Z))
    dZ = prev_grad*sigmoid*(1-sigmoid)
    return(dZ)

def forward_activation(A_previous,Weights,bias,activation="identity"):
    """
    A_previous - Activation of previous layer (size of prev layer, n_examples)
    Weights - weights matrix (size of current layer,size of prev_layer)
    bias - bias vector (size of current layer,1)
    activation - activation of current layer (identity,relu,sigmoid )(default -identity)
    """

    if(activation=="identity"):
        #Calculate Z
        Z = np.dot(A_previous,Weights) + bias
        cache = (A_previous,Weights,bias)
        return (Z,cache)
    elif(activation=="relu"):
        #Calculate Z and apply relu
        Z = np.dot(A_previous,Weights) + bias
        A_current, act_cache = relu(Z)

        cache = ((A_previous,Weights,bias),act_cache)
        return (A_current,cache)
    elif(activation=="sigmoid"):
        #Calculate Z and apply sigmoid
        Z = np.dot(A_previous,Weights) + bias
        A_current, act_cache = sigmoid(Z)

        cache = ((A_previous,Weights,bias),act_cache)
        return (A_current,cache)

def backward_activation(previous_grad,cache,activation="identity"):
    """
    previous_grad - gradient of previous layer
    cache - stored for computing the backward pass efficiently
    activation - activation of current layer (identity,relu,sigmoid )(default -identity)
    """
    if(activation=="identity"):
        #collect weights, bias and activation
        A_prev , Weights, bias = cache
        n_features = A_prev.shape[1] 
        #Calculate dW, dB and dA_prev
        dW = (1/n_features)*np.dot(A_prev.T,previous_grad)
        db = (1/n_features)*np.sum(previous_grad,axis=0,keepdims=True)
        dA_prev = np.dot(previous_grad,Weights.T)

        return(dA_prev,dW,db)
    elif(activation=="relu"):
        lin_cache , act_cache =cache
        #calculate back relu
        prev_linear_grad = back_relu(previous_grad,act_cache)

        A_prev , Weights, bias = lin_cache
        #Calculate dW, dB and dA_prev
        n_features = A_prev.shape[1] 
        dW = (1/n_features)*np.dot(A_prev.T,prev_linear_grad)
        db = (1/n_features)*np.sum(prev_linear_grad,axis=0,keepdims=True)
        dA_prev = np.dot(prev_linear_grad,Weights.T)

        return(dA_prev,dW,db)

    elif(activation=="sigmoid"):
        lin_cache , act_cache =cache
        #calculate back sigmoid
        prev_linear_grad = back_sigmoid(previous_grad,act_cache)
        
        #Calculate dW, dB and dA_prev
        A_prev , Weights, bias = lin_cache
        n_features = A_prev.shape[1] 
        dW = (1/n_features)*np.dot(A_prev.T,prev_linear_grad)
        db = (1/n_features)*np.sum(prev_linear_grad,axis=0,keepdims=True)
        dA_prev = np.dot(prev_linear_grad,Weights.T)

        return(dA_prev,dW,db)
        