import numpy as np
from utils import OptmAlgo



class LocallyWeightedRegression:

    def __init__(self, learning_rate=0.01, n_iters=1000, optm_algo = OptmAlgo.BGD.name, batch_size=32, tau = 0.08):

        self.lr = learning_rate
        self.n_iters = n_iters
        self.optm_algo = optm_algo
        self.batch_size = batch_size  
        self.tau = tau

        self.weights = None
        self.bias = None
        self.costs = []
        self._proximity_weights = None

    # since proximity_weights is calculated as a diagonal matrix,
    # decided to test properties for the purpose of accessing it as a vector
    @property # This is the getter method
    def proximity_weights(self):
        if not (self._proximity_weights is None):
            return self._proximity_weights.diagonal()

    @proximity_weights.setter # This is the setter method
    def proximity_weights(self, value):
        self._proximity_weights = value

    # Weight Functions ------------------------------
    # TODO: check if the name gaussian is correct for this kernel
    def gaussian_weight(self, X, x_new):
        """Extracts the proximity data points weights,
        using gaussian ( bell shaped ) weight function.
        NOTES: - i used proximity_weights as a diagonal matrix for calculation purpose

        :param X    : input data
        :param x_new: data point to estimate
        """
        
        # x_new = np.array(x_new)
        self.proximity_weights = np.identity(X.shape[0])

        for i in range(X.shape[0]):
            self._proximity_weights[i,i] = np.exp(-np.dot((X[i]- x_new).T, (X[i]- x_new))/(2*(self.tau**2)))

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
            dw = (1 / n_samples) * np.dot(np.dot(self._proximity_weights,X).T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # update parameters
            self.weights = self.weights- self.lr * dw
            self.bias = self.bias-self.lr * db

            self.costs.append( (1/n_samples) * np.sum( (y_predicted - y)**2 ) )

    # ------------------------------------------------

    # TODO: fitline refactoring , maybe get rid of it?

    def predict(self, X, y, x_new):
        """Predict the approximated value of x_new."""

        # reshaping target variable (m,1) to (m,) and initialization
        y = y.reshape(y.shape[0])
        
        x_new = np.array(x_new)
        x_new_shape = (x_new.shape[0], 1) if len(x_new.shape) == 1 else x_new.shape
        x_new = x_new.reshape(x_new_shape)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        predictions = []

        # estimate the result for each x_new data (non parametric model)
        for x_i in x_new:

            self.gaussian_weight(X,x_i)
            self.batch_GD(X, y)
            predictions.append(np.dot(x_i, self.weights) + self.bias)
            print(f"X_new : {x_i} -- predicted value : {np.dot(x_i, self.weights) + self.bias}")

        # only for when we have 1D data
        if X.shape[1] == 1:
            # fitline (estimations)
            fitline = [np.dot(min(X[:,0]), self.weights) + self.bias, np.dot(max(X[:,0]), self.weights) + self.bias ]
        else:
            fitline = None

        return predictions, fitline
        
        