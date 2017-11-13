# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 20:35:27 2017

@author: leemj
"""

from BA_kernel_function import polynomial_kernel
from BA_kernel_function import linear_kernel
from BA_kernel_function import rbf_kernel

import numpy as np

def kernel_matrix(X, kernel, coef0=1.0, degree=3, gamma=0.1):
    X = np.array(X,dtype=np.float64)
    mat = []
    for i in X:
        row = []
        for j in X:
            if kernel=='poly':
                row.append(polynomial_kernel(i,j,coef0,degree))
            elif kernel=='linear':
                row.append(linear_kernel(i,j,coef0))
            elif kernel=='rbf':
                row.append(rbf_kernel(i,j,gamma))
            else:
                row.append(np.dot(i,j))
        mat.append(row)
    return mat