import numpy as np
import random
from enum import Enum, auto

"""
TODO:
"""

class OptmAlgo(Enum):
    """Optimization Algorithms Available."""
    
    # Batch Gradient Descent
    BGD = auto()
    # Stochastic Gradient Descent
    # SGD = auto()
    # Mini-Batch Gradient Descent
    # MBGD = auto()

class LocallyWeightedRegression:

    def __init__(self, learning_rate=0.01, n_iters=1000, optm_algo = OptmAlgo.BGD.name, batch_size=32, tau = 0.08):

        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.optm_algo = optm_algo
        self.batch_size = batch_size  
        self.costs = []
        self.proximity_weights = None
        self.tau = tau
        self.fitline = []

    # TODO: check if the name gaussian is correct for this kernel
    # TODO: refactor this function and add a docstring
    def kernel_gaussian(self, X, x_new):
        
        x_new = np.array(x_new)
        self.proximity_weights = np.identity(X.shape[0])
        ww = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            self.proximity_weights[i,i] = np.exp(-np.dot((X[i]- x_new).T, (X[i]- x_new))/(2*(self.tau**2)))
            ww[i] = np.exp(-np.dot((X[i]- x_new).T, (X[i]- x_new))/(2*(self.tau**2)))

        return ww

    # Gradient Descent Algorithms --------------------

    def batch_GD(self, X, y):
        """Batch gradient descent.
        
        :param X: input data (m,n)
        :param y: input target variable (m,)
        """
        n_samples = X.shape[0]

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # compute gradients
            dw = (1 / n_samples) * np.dot(np.dot(self.proximity_weights,X).T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # update parameters
            self.weights = self.weights- self.lr * dw
            self.bias = self.bias-self.lr * db

            self.costs.append( (1/n_samples) * np.sum( (y_predicted - y)**2 ) )

    # ------------------------------------------------

    def predict(self, X, y, x_new):
        """Predict the approximated value of x_new."""

        # reshaping target variable (m,1) to (m,)
        y = y.reshape(y.shape[0])

        self.kernel_gaussian(X,x_new)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        self.batch_GD(X, y)
        for i in X:
            self.fitline.append( np.dot(i, self.weights) + self.bias )
        
        