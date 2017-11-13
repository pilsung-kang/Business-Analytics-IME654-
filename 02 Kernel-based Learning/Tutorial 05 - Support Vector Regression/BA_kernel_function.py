# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 20:33:06 2017

@author: leemj
"""
#BA_kernel_function.py
import numpy as np

    
def linear_kernel(x1,x2,coef0=1.0):
    return np.dot(x1,x2)+coef0

def polynomial_kernel(x1,x2,coef0=1.0,degree=3):
    return (np.dot(x1,x2)+coef0)**degree

def rbf_kernel(x1,x2,gamma=0.1):
    return np.exp(-gamma*np.linalg.norm(x1-x2)**2)

def sigmoid_kernel(x1,x2,coef0=1.0,gamma=0.1):
    return np.tanh(gamma*np.dot(x1,x2)+coef0)


def kernel_f(x1,x2,kernel = None,coef0=1.0,degree=3,gamma=0.1):
    if kernel == 'linear':
        result = np.dot(x1,x2)+coef0
    elif kernel == 'poly':
        result = (np.dot(x1,x2)+coef0)**degree
    elif kernel == 'rbf':
        result = np.exp(-gamma*np.linalg.norm(x1-x2)**2)
    elif kernel =='sigmoid':
        result = np.tanh(gamma*np.dot(x1,x2)+coef0)
    else:
        result = np.dot(x1,x2)
    
    return result