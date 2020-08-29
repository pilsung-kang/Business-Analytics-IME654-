import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def linear_kernel(x1, x2):
    
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


class SVM(object):
    
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
    
        if self.C is not None:
            self.C = float(self.C)

    def _kernel_transform(self, X_train, kernel):
        if kernel == 'linear':
            kfunc = linear_kernel
        elif kernel == 'polynomial':
            kfunc = polynomial_kernel
        elif kernel == 'gaussian':
            kfunc = gaussian_kernel
            
        self.kfunc = kfunc
        
        n_samples, n_features = X_train.shape
        self.n_samples = n_samples 
        self.n_features = n_features 
    
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = kfunc(X_train[i], X_train[j])
             
        return K

    def __solve(self, K_matrix, y_train, C):
        
        n = y_train.shape[0] 
        y_= np.diag(y_train)  # 180 by 180(1 & -1) 
        alpha = cvx.Variable(n) 
        # constraints 
        constraints = []
        for i in range(n):
            constraints += [alpha[i] >= 0, alpha[i] <= C] 
        constraints += [y_train * alpha == 0]
        self.model_obj = cvx.Maximize(np.ones(n) * alpha - .5 * cvx.quad_form(alpha, y_.dot(K_matrix).dot(y_)))#convex
        #quad form 
        self.model = cvx.Problem(objective=self.model_obj, constraints=constraints)
        self.model.solve() 
        
        
        #support vector 
        self.a = np.array(alpha.value).flatten()
        self.sv = self.a > 1e-5 
        self.ind = np.arange(len(self.a))[self.sv] 
        self.a = self.a[self.sv] 
        self.sv_y = y_train[self.sv]
        self.sv_x= self.X_train[self.sv] 
        
        # intercept
        self.b = 0 
        for n in range(len(self.a)):
            self.b+=self.sv_y[n]
            self.b-=np.sum(self.a*self.sv_y*K_matrix[self.ind[n],self.sv])
        self.b/=len(self.a)
        
        #weight vector
        if self.kernel == 'linear':
            self.w = np.zeros(self.n_features)
            for n in range(len(self.a)):
                    self.w += self.a[n] * self.sv_y[n] * self.sv_x[n]
        else:
            self.w = None
            
    
    def predict(self,X):
        if self.w is not None:
           
            return np.dot(X,self.w) + self.b
          
        else: 
            y_predict = np.zeros(len(X))#
            
            for i in range(len(X)):
                s=0
                for a, sv_y, sv_x in zip(self.a, self.sv_y, self.sv_x):
                    s += a * sv_y * self.kfunc(X[i], sv_x)
                y_predict[i] = s
                
            return y_predict + self.b 
    def signpredict(self, X_test):
        
        return np.sign(self.predict(X_test))
    
    def fit(self, X_train, y_train):
        self.X_train = X_train 
        K = self._kernel_transform(X_train, self.kernel) 
        self.K = K
        self.__solve(K, y_train, self.C)