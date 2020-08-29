# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:42:14 2017

@author: SunyoolChae
"""

import pandas as pd
import numpy as np

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']

df.head()

X = df.iloc[:,0:4].values
y = df.iloc[:,4].values

#define kernel matrix
def kernel_matrix(x, kernel=None, d=3, sigma=None, C=1.):
    
    n = x.shape[0]
    if sigma is None:
        sigma = 1./n
    
    xxt = x.dot(x.T)
    if kernel == 'polynomial':
        return (C + xxt)**d
    elif kernel == 'sigmoid':
        return np.tanh(sigma*xxt + C)
    elif kernel == 'rbf':
        A = x.dot(x.T)
        B = np.repeat(np.diag(xxt), n).reshape(n, n)
        return np.exp(-(B.T - 2*A + B)/(2*sigma**2))
    else:
        return xxt
    
K = kernel_matrix(X, kernel='polynomial', sigma=0.2)
print(K.shape)
print(K)

n = K.shape[0]
one_mat = np.repeat(1/n, n**2).reshape(n, n)
gram = K - one_mat.dot(K) - K.dot(one_mat) + one_mat.dot(K).dot(one_mat)

eigen_vals, eigen_vecs = np.linalg.eigh(gram)

print("eigen_values \n{}".format(eigen_vals))
print("eigen_vectors \n{}".format(eigen_vecs))


eigen_pairs = [(eigen_vals[i], eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key = lambda x: x[0], reverse=True)


from sklearn.decomposition import KernelPCA
kpca0 = KernelPCA(n_components=3, kernel='poly')
kpca0.fit(X)


kpca0_eigen_vecs = kpca0.alphas_
eigen_vecs_for_comparison = np.vstack([eigen_pairs[0][1], eigen_pairs[1][1], eigen_pairs[2][1]]).T

import matplotlib.pyplot as plt
%matplotlib inline

#first eigenvector
plt.subplot(2,1,1)
plt.plot(kpca0_eigen_vecs[:,0], c='blue')
plt.subplot(2,1,2)
plt.plot(eigen_vecs_for_comparison[:,0], c='green')

#Second eigenvector
plt.subplot(2,1,1)
plt.plot(kpca0_eigen_vecs[:,1], c='blue')
plt.subplot(2,1,2)
plt.plot(eigen_vecs_for_comparison[:,1], c='green')

#Third eigenvector
plt.subplot(2,1,1)
plt.plot(kpca0_eigen_vecs[:,2], c='blue')
plt.subplot(2,1,2)
plt.plot(eigen_vecs_for_comparison[:,2], c='green')

# transform data
n_components = 2

transformed_data = []
for j in range(n):
    loc = np.zeros(n_components)
    for k in range(n_components):
        inner_prod_sum = 0.
        for i in range(n):
            inner_prod_sum += eigen_pairs[k][1][i] * gram[j,i]
        loc[k] = inner_prod_sum/np.sqrt(eigen_pairs[k][0])
    transformed_data.append(loc)    
    
transformed_data = np.array(transformed_data)

label = df['class'].unique()
print(label)

with plt.style.context("seaborn-darkgrid"):
    for l in zip(label):
        plt.scatter(transformed_data[y==l,0], transformed_data[y==l,1],
                    label=l)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.show()
    
kpca0 = KernelPCA(n_components=2, kernel='poly')
Y = kpca0.fit_transform(X)

with plt.style.context("seaborn-darkgrid"):  
    for l in label:
        plt.scatter(Y[y==l,0], Y[y==l,1],label=l)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.show()
    
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Y_ = pca.fit_transform(X)

with plt.style.context("seaborn-darkgrid"):  
    for l in label:
        plt.scatter(Y_[y==l,0], Y_[y==l,1],label=l)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.show()
