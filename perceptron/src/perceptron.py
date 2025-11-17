# Start building perceptron model
import numpy as np

class Perceptron:

    def __init__(self, eta, n_iter, random_state = 42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_


    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


    def fit(self, X, y):
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.float64(0.0)
        self.resid_ = []

        for _ in range(self.n_iter):
            resids = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                resids += int(update != 0.0)
            self.resid_.append(resids)
        
        return self