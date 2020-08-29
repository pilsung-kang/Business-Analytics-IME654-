import numpy as np
from scipy.sparse.linalg import eigsh

def LLE(X, n_neighbors, n_components):

    x_matrix = np.asmatrix(X).T
    D, N = x_matrix.shape

    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    distance = np.asmatrix(D)

    index=np.argsort(distance,axis=0)
    neighborhood = index[1:(n_neighbors+1),:]

    # 30 x 64
    W = np.zeros((n_neighbors,N))
    tol = 1e-3


    # for variables
    for i in range(N):
        z = x_matrix[:, np.array(neighborhood[:,i]).reshape(-1)] - np.tile(x_matrix[:,i],(1,n_neighbors)) # shift ith pt to origi
        C = z.T*z # local covariance
        C = C + np.eye(n_neighbors) * tol * np.trace(C)
        I = np.ones((n_neighbors))
        W[:,i] = np.linalg.solve(np.array(C),I)
        W[:,i] = W[:,i]/sum(W[:,i])


    M = np.eye(N)  # M = (I-W)' (I-W)

    for i in range(N):
        w = W[:, i]
        j = neighborhood[:, i]
        M[i, j] = M[i, j] - np.asmatrix(w).T
        M[j, i] = M[j, i] - np.asmatrix(w).T
        M[j, j] = M[j, j] + np.asmatrix(w*w.T).T

    eigenvals, Y = eigsh(M, n_components + 1, tol=1e-3, sigma=0.0, maxiter = 100)

    return Y[:, 1:], np.sum(eigenvals[1:])
