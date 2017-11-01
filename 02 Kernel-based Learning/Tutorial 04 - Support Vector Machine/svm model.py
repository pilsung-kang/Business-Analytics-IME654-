# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:59:24 2017

@author: Yooyeon
"""

import numpy as np
import cvxpy as cvx


def linear_kernel(x1, x2):
    
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=2.0):
    
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
            
        self.kfunc = kfunc# 선택된 Kenel을 모델에 저장 
        
        n_samples, n_features = X_train.shape
        self.n_samples = n_samples # 모델의 관측치 수 저장
        self.n_features = n_features # 모델의 변수 수 저장
        
        # Kernel Matrix를 만든다
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = kfunc(X_train[i], X_train[j])
                
        return K#변형된 K

    def __solve(self, K_matrix, y_train, C):
        
        n = y_train.shape[0] # y_train의 갯수만 
        y_= np.diag(y_train)  # 180 by 180 대각선 (1 & -1) , K_matrix(180,180)과의 내적을 구하기 위해 맞춰줌 
        # primal -> dual ->풀기위해 cvxpy 를 사용 (convec optimization problem)
        #cvxpy : convex optimization problem 을 위한 모델링-> 변수, convex식, 변수에 대한 제약식-> Problem(convex 식, 변수제약식 )
        # cvxpy를 통해 alpha 를 구한다 ,dual 식을 풀기위해  alpha를 찾는게 관건이다
        alpha = cvx.Variable(n) 
        # constraints 
        constraints = []
        for i in range(n):
            constraints += [alpha[i] >= 0, alpha[i] <= C]  # alpha 제약식 
        constraints += [y_train * alpha == 0]
        #model의 최적 식과 식의 결과값 
        self.model_obj = cvx.Maximize(np.ones(n) * alpha - .5 * cvx.quad_form(alpha, y_.dot(K_matrix).dot(y_)))#convex식 
        #quad form 
        self.model = cvx.Problem(objective=self.model_obj, constraints=constraints)
        self.model.solve() # max값 
        
        
        #support vector 
        self.a = np.array(alpha.value).flatten()# alpha 값을 2-dimension -> 1-dimension
        self.sv = self.a > 1e-5 # true or false (0보다 크다) sv에 대응하는 alpha 값을 sv에 저장 
        self.ind = np.arange(len(self.a))[self.sv] #alpha 값 중에 sv에 대응하는 alpha 배열 
        self.a = self.a[self.sv] #a 변수 = sv에 대응하는 alpha 값들 저장
        self.sv_y = y_train[self.sv]#  sv_y = train data 중 sv에 대응하는 y 저장
        self.sv_x= self.X_train[self.sv] # sv_y = train data 중 sv에 대응하는 y 저장 
        
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
            
    
    def predict(self,X):# X=np.c_[xx.ravel(), yy.ravel()]: train_X와 test_X를 합친거에 대한 예측값 -> 전체 X값
        if self.w is not None:
           
            return np.dot(X,self.w) + self.b # wxT+b식 활용 
          
        else: 
            y_predict = np.zeros(len(X))# 전체 X에 대한 전체 Y값 예측 
            
            for i in range(len(X)):
                s=0
                for a, sv_y, sv_x in zip(self.a, self.sv_y, self.sv_x):
                    s += a * sv_y * self.kfunc(X[i], sv_x)#wxT식 <- x 대신 kernel이 들어간 식 활용
                y_predict[i] = s
                
            return y_predict + self.b #w*kernel(x)+b 활용
    
    def signpredict(self, X_test):
        
        return np.sign(self.predict(X_test))#y_test의 부호 
    
    def fit(self, X_train, y_train):
        self.X_train = X_train # 모델 에  X_train 저장 
        #K: 변형된 kernel matrix 
        K = self._kernel_transform(X_train, self.kernel) # Kenel 함수를 선택(X_train을 Kernel 함수에 넣는다)
        self.K = K
        self.__solve(K, y_train, self.C)

