import pandas as pd
import numpy as np
import matplotlib as plt

class PCA_jay:
    def __init__(self, input_data, label=0):
        self.data = input_data
        self.label = label
    
    def eigen(self):
        self.X_std = (self.data - np.mean(self.data, axis=0)) / np.std(self.data, axis=0)
        mean_vec = np.mean(self.X_std, axis=0)
        cov_mat = (self.X_std - mean_vec).T.dot((self.X_std - mean_vec)) / (self.X_std.shape[0]-1)
        self.eig_vals, self.eig_vecs = np.linalg.eig(cov_mat)
        return self.eig_vals, self.eig_vecs
    
    def projection(self, n_components=2):
        eig_pairs = [(np.abs(self.eig_vals[i]), self.eig_vecs[:,i]) for i in range(len(self.eig_vals))]
        eig_pairs.sort(key = lambda x: x[0], reverse=True)
        projection_matrix  = np.hstack((eig_pairs[i][1].reshape(len(self.eig_vals),1)) for i in range(n_components))
        self.Y = self.X_std.dot(projection_matrix)
        return self.Y
    
    def scree_plot(self):
        total = sum(self.eig_vals)
        explained_var = [(i/total) * 100 for i in sorted(self.eig_vals, reverse=True)]
        cum_var = np.cumsum(explained_var)

        with plt.style.context('seaborn-darkgrid'):
            
            plt.bar(range(len(explained_var)), explained_var, align='center',
                    label='individual explained variance')
            plt.step(range(len(explained_var)), cum_var, where='mid',
                     label='cumulative explained variance',color = "red")
            plt.ylabel("Cumulative variance")
            plt.xlabel("Principal components")
            plt.tight_layout()
            plt.legend(loc='best')
            plt.show()
            
    def pca_plot(self):
        label = np.unique(self.label)
        with plt.style.context("seaborn-darkgrid"):  
            for l in label:
                plt.scatter(self.Y[y==l,0], self.Y[y==l,1],label=l)
            plt.xlabel("PC 1")
            plt.ylabel("PC 2")
            plt.legend()
            plt.show() 



