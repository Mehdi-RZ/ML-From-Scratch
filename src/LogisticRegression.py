import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.01, n_iters=1000):

        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.costs = []


    def batch_GD(self, X, y):
        """Batch gradient descent.
        
        :param X: input data (m,n)
        :param y: input target variable (m,)
        """
        n_samples = X.shape[0]

        for _ in range(self.n_iters):
            
            
            z = X.dot(self.weights)
            y_predicted = (1/(1+np.exp(-z)))

            dw = (1/n_samples) * X.T.dot(y_predicted - y)
            db = (1/n_samples) * (y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.costs.append( (1/n_samples) * np.sum( (y_predicted - y)**2 ) )


    def fit(self, X, y):
        """Adjusting Weights to fit the Data."""

        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        self.batch_GD(X, y)
    

    def predict(self, X):
        """Predict the approximated values of the target variable.
        
        :param X : new data used to predict new values for the target variable
        :return: predicted values of the target variable for the X new data
        """
        z = X.dot(self.weights)
        y_predicted = (1/(1+np.exp(-z)))

        return y_predicted


    