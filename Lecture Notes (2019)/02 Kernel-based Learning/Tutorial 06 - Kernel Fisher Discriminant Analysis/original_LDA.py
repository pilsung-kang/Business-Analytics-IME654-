# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 16:45:39 2017

@author: dmqm171
"""

# ----- Linear discriminant analysis -----
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Data Load & Preprocessing
data = pd.read_csv(os.path.join(os.path.join(os.path.abspath(''), os.pardir), 'C:/Users/dmqm171/Documents/R/data/Datasets_KFDA/abalone.csv'))
data = data.iloc[0:100,:]
df = pd.DataFrame.as_matrix(data)
y = df[:,0]
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1
label_dict = {1: 'A', 2: 'I'}
X = df[:,1:]
nrow = X.shape[0]
ncol = X.shape[1]
cg = np.unique(y)
for i in range(X.shape[0]):
    if y[i] == cg[0]:
        y[i] = 1
    else:
        y[i] = 2

y = y.astype(int)


feature_dict = {i:label for i,label in zip(
                range(ncol),
                  ('Length',
                  'Diameter',
                  'Height',
                  'Whole_Weiht',
                  'Shucked_Weight',
                  'Viscera_Weight',
                  'Shell_Weight',
                  'Rings'))}
    
# Histograms and feature selection
# Just to get a rough idea how the samples of our three classes ω1ω1, ω2ω2 are distributed, 
# let us visualize the distributions of the four different features in 1-dimensional histograms.

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12,6))
for ax,cnt in zip(axes.ravel(), range(ncol)):  
    # set bin sizes
    min_b = math.floor(np.min(X[:,cnt]))
    max_b = math.ceil(np.max(X[:,cnt]))
    bins = np.linspace(min_b, max_b, 25)

    # plottling the histograms
    for lab,col in zip(range(1,3), ('blue', 'red')):
        ax.hist(X[y==lab, cnt],
                   color=col,
                   label='class %s' %label_dict[lab],
                   bins=bins,
                   alpha=0.5,)
    ylims = ax.get_ylim()

    # plot annotation
    leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)
    ax.set_ylim([0, max(ylims)+2])
    ax.set_xlabel(feature_dict[cnt])
    ax.set_title('Abalon histogram #%s' %str(cnt+1))

    # hide axis ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

axes[0][0].set_ylabel('count')
axes[1][0].set_ylabel('count')
fig.tight_layout()       
plt.show()



# Step 1 of LDA: Computing the d-dimensional mean vectors
np.set_printoptions(precision=4)
mean_vectors = []
for cl in range(1,3):
    mean_vectors.append(np.mean(X[y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))
    
# Step 2 of LDA: computing the Scatter matrices S_W, S_B
# within-class scatter matrix Sw
S_W = np.zeros((ncol,ncol))
for cl,mv in zip(range(1,3), mean_vectors):
    class_sc_mat = np.zeros((ncol,ncol))                  # scatter matrix for every class
    for row in X[y == cl]:
        row, mv = row.reshape(ncol,1), mv.reshape(ncol,1) # make column vectors
        class_sc_mat += ((row-mv).dot((row-mv).T)).astype('float64')
    S_W += class_sc_mat                             # sum class scatter matrices
print('within-class Scatter Matrix:\n', S_W)

# Between-class scatter matrix Sb
overall_mean = np.mean(X, axis=0)
S_B = np.zeros((ncol,ncol))
for i,mean_vec in enumerate(mean_vectors):  
    n = X[y==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(ncol,1) # make column vector
    overall_mean = overall_mean.reshape(ncol,1) # make column vector
    S_B += (n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)).astype('float64')
print('between-class Scatter Matrix:\n', S_B)

#Step 3 of LDA: Solving the generalized eigenvalue problem for the matrix SW−1SB
#:Next, we will solve the generalized eigenvalue problem 
# for the matrix S−1WSBSW−1SB to obtain the linear discriminants.

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(ncol,1)   
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))

#Checking the eigenvector-eigenvalue calculation
#A quick check that the eigenvector-eigenvalue calculation is correct and satisfy the equation:
#Av=λv #where #A=S−1WSB #v=Eigenvector #λ=Eigenvalue

for i in range(len(eig_vals)):
    eigv = eig_vecs[:,i].reshape(ncol,1)
    np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
                                         eig_vals[i] * eigv,
                                         decimal=6, err_msg='', verbose=True)
print('ok')


# Step 4 of LDA: Selecting linear discriminants for the new feature subspace
# 4.1 sorting the eigenvectors by decreasing eigenvalues
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])
    
# 4.2 Choosing k eigenvectors with the largest eigenvalues
W = np.hstack((eig_pairs[0][1].reshape(ncol,1), eig_pairs[1][1].reshape(ncol,1)))
W = W.astype('float32')
print('Matrix W:\n', W.real)
X_lda = X.dot(W)


#assert X_lda.shape == (150,2), "The matrix is not 150x2 dimensional."
#assert X_lda.shape == (150,2), "The matrix is not 150x2 dimensional."
from matplotlib import pyplot as plt
def plot_step_lda():

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,3),('^', 's'),('blue', 'red')):

        plt.scatter(
                x=X_lda[:,0].real[y == label],
                y=X_lda[:,1].real[y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Abalon projection onto the first 2 linear discriminants')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()

plot_step_lda()

























#assert X_lda.shape == (150,2), "The matrix is not 150x2 dimensional."
from matplotlib import pyplot as plt

def plot_step_lda():

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,3),('^', 's', ),('blue', 'red')):

        plt.scatter(x=X_lda[:,0].real[y == label],
                y=X_lda[:,0].real[y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )

    plt.xlabel('KLD1')
    plt.ylabel('KLD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('KLDA: Abalon projection onto the first 2 linear discriminants')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()

plot_step_lda()



