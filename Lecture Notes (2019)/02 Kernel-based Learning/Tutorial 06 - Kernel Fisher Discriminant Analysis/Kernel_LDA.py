# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# ----- Linear discriminant analysis -----
# ----- abalone KFDA -----
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

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

# ----- Define kernel & Transform function  ----- 
#Kernel function
def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def rbf_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def sigmoid_kernel(x, y, kappa, C):
    return np.tanh(kappa * np.dot(x,y) + C)

#transform original to kernel
def kernel_transform(X, kernel):
    if kernel == 'linear':
        kfunc = linear_kernel
    elif kernel == 'polynomial':
        kfunc = polynomial_kernel
    elif kernel == 'rbf':
        kfunc = rbf_kernel
    elif kernel == 'sigmoid':
        kfunc = sigmoid_kernel
    n_samples, n_features = X.shape
    n_samples = n_samples 
    n_features = n_features 
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = kfunc(X[i], X[j])            
    return K

# ----- Get a kernel matrix -----
K_mat = kernel_transform(X, 'linear')
print (K_mat.shape)


# Histograms and feature selection
#Just to get a rough idea how the samples of our three classes ω1ω1, ω2ω2 are distributed, 
#let us visualize the distributions of the four different features in 1-dimensional histograms.

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,6))
for ax,cnt in zip(axes.ravel(), range(ncol)):  
    # set bin sizes
    min_b = math.floor(np.min(K_mat[:,cnt]))
    max_b = math.ceil(np.max(K_mat[:,cnt]))
    bins = np.linspace(min_b, max_b, 25)

    # plottling the histograms
    for lab,col in zip(range(1,3), ('blue', 'red')):
        ax.hist(K_mat[y==lab, cnt],
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
    mean_vectors.append(np.mean(K_mat[y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))
# Step 2 of LDA: computing the Scatter matrices S_W, S_B
# within-class scatter matrix
S_W = np.zeros((nrow ,nrow))
for cl,mv in zip(range(1,3), mean_vectors):
    class_sc_mat = np.zeros((nrow,nrow))                  # scatter matrix for every class
    for row in K_mat[y == cl]:
        row, mv = row.reshape(nrow,1), mv.reshape(nrow,1) # make column vectors
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat                             # sum class scatter matrices
print('within-class Scatter Matrix:\n', S_W)

# Between-class scatter matrix Sb
overall_mean = np.mean(K_mat, axis=0)
S_B = np.zeros((nrow,nrow))
for i,mean_vec in enumerate(mean_vectors):  
    n = K_mat[y==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(nrow,1) # make column vector
    overall_mean = overall_mean.reshape(nrow,1) # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
print('between-class Scatter Matrix:\n', S_B)

# ----- Find N, M, Alpha, b -----
n = nrow
idx1 = np.where(data.iloc[:,0]=='A')[0]
idx2 = np.where(data.iloc[:,0]=='I')[0]
n1 = idx1.shape[0]
n2 = idx2.shape[0]
K1, K2 = K_mat[:, idx1], K_mat[:, idx2]
N1 = np.dot(np.dot(K1, np.eye(n1) - (1 / float(n1))), K1.T)
N2 = np.dot(np.dot(K2, np.eye(n2) - (1 / float(n2))), K2.T)

#N
N = N1 + N2 + np.diag(np.repeat(0.001, n))

Ni = np.linalg.inv(N)
m1 = np.sum(K1, axis=1) / float(n1)
m2 = np.sum(K2, axis=1) / float(n2)
d = (m1 - m2)

#M
M = np.dot(d.reshape(-1, 1), d.reshape(1, -1))

#alpha
alpha = np.linalg.solve(N, d)
#alpha
b = - np.dot(alpha, (n1 * m1 + n2 * m2) / float(n))


# ----- comparare  yoonsang's Alpha and package's Alpha -----
import mlpy # mlpy is must be installed with command line function (pip, easy_install)
# Using linear kernel
Kl = mlpy.kernel_linear(X, X) # compute the kernel matrix
linear_kfda = mlpy.KFDA(lmb=0.001)
linear_kfda.learn(Kl, y) # compute the tranformation vector
zl = linear_kfda.transform(Kl) # embedded x into the kernel fisher space

# Using Gaussian kernel
sig = 1
Kg = mlpy.kernel_gaussian(X, X, sigma=sig) # compute the kernel matrix
gaussian_kfda = mlpy.KFDA(lmb=0.001)
gaussian_kfda.learn(Kg, y) # compute the tranformation vector
zg = gaussian_kfda.transform(Kg) # embedded x into the kernel fisher space
gaussian_kfda._coeff # alpha

# Using sigmoid kernel
gam=0.1
Ks = mlpy.kernel_sigmoid(X, X, gamma=gam, b=1.0) # compute the kernel matrix
sigmoid_kfda = mlpy.KFDA(lmb=0.001)
sigmoid_kfda.learn(Ks, y) # compute the tranformation vector
zs = sigmoid_kfda.transform(Ks) # embedded x into the kernel fisher space
sigmoid_kfda._coeff

# Using polynomial kernel
gam = 1.0
Kp = mlpy.kernel_polynomial(X, X, gamma=gam, b=1.0, d=2.0) # compute the kernel matrix
polynomial_kfda = mlpy.KFDA(lmb=0.001)
polynomial_kfda.learn(Kp, y) # compute the tranformation vector
zp = polynomial_kfda.transform(Kp) # embedded x into the kernel fisher space
polynomial_kfda._coeff

# Step 3 of LDA: Solving the generalized eigenvalue problem for the matrix SW−1SB
# Next, we will solve the generalized eigenvalue problem 
# for the matrix S−1W SB SW−1 SB to obtain the linear discriminants.
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(nrow,1)   
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))

#Checking the eigenvector-eigenvalue calculation
#A quick check that the eigenvector-eigenvalue calculation is correct and satisfy the equation:
#Av=λv #where #A=S−1WSB #v=Eigenvector #λ=Eigenvalue

for i in range(len(eig_vals)):
    eigv = eig_vecs[:,i].reshape(nrow,1)
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
W = np.hstack((eig_pairs[0][1].reshape(nrow,1), eig_pairs[1][1].reshape(nrow,1)))
print('Matrix W:\n', W.real)
X_lda = K_mat.dot(W)
K_mat.dot

# Now, let’s express the “explained variance” as percentage:
print('Variance explained:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))


# Step 5: Transforming the samples onto the new subspace
X_lda = K_mat.dot(W)
X_lda[:,0].shape
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

    plt.xlabel('KLD1')
    plt.ylabel('KLD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')

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



