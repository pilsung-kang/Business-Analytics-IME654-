
# -*- coding: utf-8 -*-

"""

Created on Tue Oct 31 13:58:01 2017



@author: Yooyeon

"""

import numpy as np

from sklearn.preprocessing import StandardScaler

# create 2-dimension data (class: y=1,-1)

# 100 samples from normal distribution

# linearly-separable, non-linearly-separable, linearly-separable but overlapping data



def gen_lin_separable_data():

    """Generate linearly seperable data."""

    mean1 = np.array([0,2])

    mean2 = np.array([2,0])

    cov = np.array([[0.8,0.6],[0.6,0.8]])

    X1 = np.random.multivariate_normal(mean1, cov, 100)  

    y1 = np.ones(len(X1))  

    X2 = np.random.multivariate_normal(mean2, cov, 100)

    y2 = np.ones(len(X2))*-1

    

    return X1, y1, X2, y2



def gen_non_lin_separable_data():

    """Generate non-linearly seperable data."""

    mean1 = [-1,2]

    mean2 = [1,-1]

    mean3 = [4,-4]

    mean4 = [-4,4]

    cov = [[1.0,0.8],[0.8,1.0]]

    

    X1 = np.random.multivariate_normal(mean1, cov, 50)

    X1 = np.vstack((X1, np.random.multivariate_normal(mean3,cov,50)))

    y1 = np.ones(len(X1))

    X2 = np.random.multivariate_normal(mean2, cov, 50)

    X2 = np.vstack((X2,np.random.multivariate_normal(mean4,cov,50)))

    y2 = np.ones(len(X2))*-1

    

    return X1, y1, X2, y2



def gen_lin_separable_overlap_data():

    """Generate linearly seperable but overlapping data.""" 

    mean1 = np.array([0,2])

    mean2 = np.array([2,0])

    cov = np.array([[1.5, 1.0],[1.0,1.5]])

    

    X1 = np.random.multivariate_normal(mean1, cov, 100) 

    y1 = np.ones(len(X1)) 

    X2 = np.random.multivariate_normal(mean2, cov, 100)

    y2 = np.ones(len(X2))*-1

    

    return X1, y1, X2, y2



def split_train(X1,y1,X2,y2):

    #combines the train and test datas: creates X_train, X_test, y_train, Y_test

    scaler = StandardScaler()

    

    X1_train = X1[:90]

    y1_train = y1[:90]

    X2_train = X2[:90]

    y2_train = y2[:90]

    X_train = np.vstack((X1_train, X2_train))

    y_train = np.hstack((y1_train, y2_train))

    

    X1_test = X1[90:]

    y1_test = y1[90:]

    X2_test = X2[90:]

    y2_test = y2[90:]

    X_test = np.vstack((X1_test,X2_test))

    y_test = np.hstack((y1_test,y2_test))

    

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)



    return X_train, y_train, X_test, y_test 