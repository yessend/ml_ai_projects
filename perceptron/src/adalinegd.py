import numpy as np

class AdalineGD:
    
    """
    X: input matrix, has shape [n_examples, n_features]
    y: true target, has shape [n_examples] 
    """

    def __init__(self, eta, n_iter, random_state = 42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    
    def net_input(self, X):
    # Here, since we will use net_input(X) as input
    # output is basically a vector of shape [n_examples] (raw prediction)
        return np.dot(X, self.w_) + self.b_
    

    def activation(self, X):
    # Here, since we will use net_input(X) as input
    # X is basically a vector of shape [n_examples] (raw prediction)
        return self.net_input(X)
    

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1.0, 0.0)
    

    def fit(self, X, y):

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.float64(0.0)
        self.resid_ = []


        for i in range(self.n_iter):
            forw_pass = self.activation(X)
            errors = (y - forw_pass)             # vector of erros: shape is [n_examples]
            self.w_ += self.eta * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * errors.mean()
            self.resid_.append( (errors**2).mean() )
        
        return self
