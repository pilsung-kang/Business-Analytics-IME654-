import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MDS_jay:
    def __init__(self, X, y=0, k=2, category=False):
        self.X = X
        self.k = k
        self.label = y
        self.category = category
    def distance_mat(self):
        self.dists = np.zeros((len(self.X),len(self.X)))
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                self.dists[i,j] = np.sqrt(np.sum((self.X[i,:]-self.X[j,:])**2))
        return self.dists
    def mds(self):
        n = len(self.dists)
        H = np.eye(n) - np.ones((n, n))/n
        B = -H.dot(dists**2).dot(H)/2
        eig_vals,eig_vecs = np.linalg.eigh(B)
        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:,idx]
        w = np.where(eig_vals > 0)
        L = np.diag(np.sqrt(eig_vals[w]))
        V = eig_vecs[:,w]
        V = np.squeeze(V)
        Y = V.dot(L)
        self.Y = Y[:,:self.k]
        return self.Y
    def mds_plot(self):
        if self.category == True:
            label = np.unique(self.label)
            with plt.style.context("seaborn-darkgrid"):
                for l in label:
                    plt.scatter(Y[y==l,0], Y[y==l,1],label=l)
                plt.xlabel("dimension 1")
                plt.ylabel("dimension 2")
                plt.legend()
                plt.show() 
        else:
            label = [1]
            with plt.style.context("seaborn-darkgrid"):
                plt.scatter(Y[:,0], Y[:,1])
                plt.xlabel("dimension 1")
                plt.ylabel("dimension 2")
                plt.show() 
                