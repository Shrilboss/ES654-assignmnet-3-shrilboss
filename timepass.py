from numpy import argmax
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
y = np.array([1,0,2,3,1,4,5,1,2])
y = y.reshape(len(y),1)
one = OneHotEncoder(sparse=False)
# r = one.fit_transform(y)
# print(r)
X = np.array([[2,2,2],[2,2,2],[2,2,2]])
theta = np.array([[3,1],[3,1],[3,1]])
print(np.exp(np.dot(X[0],theta)))
print(np.dot(X[0],theta[:,0]),np.dot(X[0],theta[:,1]))
print(np.exp(np.dot(X[0],theta)).sum(axis=-1,keepdims=True))
# data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
# values = np.array(data)
# print(values)
# # integer encode
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# print(integer_encoded)
# # binary encode
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded)
# # invert first example
# inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
# print(inverted)