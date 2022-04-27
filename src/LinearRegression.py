import numpy as np
from enum import Enum, auto
from utils import shuffled_seq_index

"""
TODO:
"""

class OptmAlgo(Enum):
    """Optimization Algorithms Available."""
    
    # Batch Gradient Descent
    BGD = auto()
    # Stochastic Gradient Descent
    SGD = auto()
    # Mini-Batch Gradient Descent
    MBGD = auto()


class LinearRegression:

    def __init__(self, learning_rate=0.01, n_iters=1000, optm_algo = OptmAlgo.BGD.name, batch_size=32):

        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.optm_algo = optm_algo
        self.batch_size = batch_size
        self.costs = []
        

    # Gradient Descent Algorithms --------------------

    def batch_GD(self, X, y):
        """Batch gradient descent.
        
        :param X: input data (m,n)
        :param y: input target variable (m,)
        """

        print("Running batch G.D...")
        n_samples = X.shape[0]

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            self.costs.append( (1/n_samples) * np.sum( (y_predicted - y)**2 ) )


    def stochastic_GD(self, X, y):
        """Stochastic gradient descent.
        data shapes: X : (m,n)  //  y : (m,)  //  X_i : (n,)  
        y_i : ()  //  y_pred : ()  //  Weight : (n,)
        """

        print("Running stochastic G.D...")
        n_samples = X.shape[0]
        # create a list of shuffled indexes
        idx = shuffled_seq_index(X.shape[0], self.n_iters)
        
        for i in range(self.n_iters):

            y_predicted = np.dot(X[idx[i]].T, self.weights) + self.bias

            # compute gradients
            dw = (1 / n_samples) * np.dot(X[idx[i]].T, (y_predicted - y[idx[i]]))
            db = (1 / n_samples) * np.sum(y_predicted - y[idx[i]])
            
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
            cost = (1/n_samples)* np.square(y_predicted-y[idx[i]])
            # self.costs.append(cost)
            # every 100th iteration record the cost
            if i%100==0: 
                self.costs.append(cost)


    def mini_batch_GD(self, X, y):
        """Mini-Batch gradient descent.       

        :param X: input data (m,n)
        :param y: input target variable (m,)
        """

        print("Running mini-batch G.D ...")
        n_samples = X.shape[0]
        idx = shuffled_seq_index(X.shape[0], self.n_iters*self.batch_size)
        start_idx = 0

        for i in range(self.n_iters):

            end_idx = start_idx+self.batch_size
            y_predicted = np.dot(X[idx[start_idx:end_idx]], self.weights) + self.bias

            # compute gradients
            dw = (1 / n_samples) * np.dot(X[idx[start_idx:end_idx]].T, (y_predicted - y[idx[start_idx:end_idx]]))
            db = (1 / n_samples) * np.sum(y_predicted - y[idx[start_idx:end_idx]])
        
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
            cost =  (1/n_samples)*np.sum(np.square(y_predicted-y[idx[start_idx:end_idx]]))
            start_idx += self.batch_size

            # self.costs.append(cost)   

            if i%20==0: # at every 20th iteration record the cost
                self.costs.append(cost)        

    # ------------------------------------------------

    def fit(self, X, y):
        """Adjust and fit parameters to the data."""

        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # reshaping target variable (m,1) to (m,)
        y = y.reshape(y.shape[0])

        print(f"X --> shape: {X.shape}")
        print(f"y--> shape: {y.shape}")
        print(f"THETA --> shape: {self.weights.shape}")

        # optimisation algorithm choice
        if self.optm_algo == OptmAlgo.BGD.name:
            self.batch_GD(X, y)
        
        elif self.optm_algo == OptmAlgo.SGD.name:
            self.stochastic_GD(X, y)
        
        elif self.optm_algo == OptmAlgo.MBGD.name:
            self.mini_batch_GD(X, y)
        

    def predict(self, X):
        """Predict the approximated values of the target variable.
        
        :param X : new data used to predict new values for the target variable
        :return: predicted values of the target variable for the X new data
        """

        y_approximated = np.dot(X, self.weights) + self.bias

        return y_approximated
