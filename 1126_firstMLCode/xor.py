# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 20:07:59 2018
BPNN: Xor
@author: mhy
"""
import numpy as np

# true value for input and output
x0 = np.array([[1,1],[1,0],[0,1],[0,0]])
d = np.array([0.0, 1.0, 1.0, 0.0])

# initial value for input, layer, output
x = np.c_[x0, [1,1,1,1]]   #column combine with bias
y = np.zeros([4,3])
o = np.array([0.0, 0.0, 0.0, 0.0])

# initial value for weight number
v = np.array([[1.0, 1.0],[-1.0, -1.0],[0.0, 0.0]])    #first layer
w = np.array([[1.0],[-1.0],[0.0]])                    #second layer

# learning rate
lr = 0.6

# initial error
err = 0.5

# activate function, sigmoid
def sigmoid(x):
    y = 1.0/(1.0+np.exp(-x))
    return y

step = 0
# main 
while err >= 0.008: 
    step = step + 1
    err = 0    
    for i in range(4):
        y[i, 0] = sigmoid(np.matmul(x[i,:], v[:,0]))
        y[i, 1] = sigmoid(np.matmul(x[i,:], v[:,1]))
        y[i, 2] = 1    #bias
        o[i] = sigmoid(np.matmul(y[i, :], w))
        
        err = err + 0.5 * np.square(o[i] - d[i])
        
        # back propagation
        g = o[i]*(1 - o[i])*(d[i] - o[i])
        
        e1 = g * w[0] * y[i, 0] * (1 - y[i, 0])
        e2 = g * w[1] * y[i, 1] * (1 - y[i, 1])
        
        # update weights
        w[0] = w[0] + lr * g * y[i, 0]
        w[1] = w[1] + lr * g * y[i, 1]
        w[2] = w[2] + lr * g * y[i, 2]
        
        v[:, 0] = v[:, 0] + lr * e1 * x[i, :]
        v[:, 1] = v[:, 1] + lr * e2 * x[i, :]

print('error is now:', err)
print('total step is:', step)
print('output is:\n', o)
