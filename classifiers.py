import numpy as np
from kernels import *



class KernelRidge():
    '''
    Kernel Ridge Regression
    
    Methods
    ----
    fit
    predict
    '''
    def __init__(self, sigma=None, lambd=0.1):
        self.kernel = rbf_kernel
        self.sigma = sigma
        self.lambd = lambd

    def fit(self, X, y):
        n, p = X.shape
        assert (n == len(y))
    
        self.X_train = X
        
        # Compute default sigma from data
        if self.sigma is None:
            self.sigma = sigma_from_median(X)
        
        A = self.kernel(X, X, sigma = self.sigma) + n * self.lambd * np.eye(n)
        
        ## self.alpha = (K + n lambda I)^-1 y
        # Solution to A x = y
        self.alpha = np.linalg.solve(A , y)

        return self
        
    def predict(self, X):
        # Prediction rule: 
        K_x = self.kernel(X, self.X_train, sigma=self.sigma)
        return K_x @ self.alpha


class KernelRidge_polynomial():
    '''
     Kernel Ridge Regression
        
    Methods
    ----
    fit
    predict
    '''
    def __init__(self):
        self.kernel = polynomial_kernel

    def fit(self, X, y):
        n, p = X.shape
        assert (n == len(y))
    
        self.X_train = X
        
        A = self.kernel(X, X, degree = 3) * np.eye(n)
        
        ## self.alpha = (K + n lambda I)^-1 y
        # Solution to A x = y
        self.alpha = np.linalg.solve(A , y)

        return self
        
    def predict(self, X):
        # Prediction rule: 
        K_x = self.kernel(X, self.X_train, degree = 3)
        return K_x @ self.alpha