import numpy as np

class AdalineGD:
    
    """
    ADAptive LInear NEuron classifier with full batch Gradient Descent.
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
        n_iter : int
    Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
   
    
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    resid_ : list
        Mean squared error loss function values in each epoch.
    
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
        return X
    

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1.0, 0.0)
    
    """
    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples
        is the number of examples and
        n_features is the number of features.
    
    y : array-like, shape = [n_examples]
        Target values.
    
    
    Returns
    -------
    self : object
    """
    
    def fit(self, X, y):

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.float64(0.0)
        self.resid_ = []


        for i in range(self.n_iter):
            forw_pass = self.activation(self.net_input(X))
            errors = (y - forw_pass)             # vector of erros: shape is [n_examples]
            self.w_ += self.eta * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * errors.mean()
            self.resid_.append( (errors**2).mean() )
        
        return self




class AdalineSGD:

    """
    ADAptive LInear NEuron classifier with Stochastic Gradient Descent.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent
        cycles.
    random_state : int
        Random number generator seed for random weight
        initialization.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    losses_ : list
        Mean squared error loss function value averaged over all
        training examples in each epoch.
    """

    def __init__(self, eta, n_iter, shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
    

    def _initialize_weights(self, m):
        # Initialize weights and set the w_initialized flag to True
        self.rgen = np.random.RandomState(self.random_state) # create a random number generator
        self.w_ = self.rgen.normal(loc = 0.0, scale = 0.01, size = m)
        self.b_ = np.float64(0.0)
        self.w_initialized = True


    def _shuffle(self, X, y):
        # Shuffle the training data
        r = self.rgen.Permutation(len(y))
        return X[r], y[r]
    

    def _update_weights(self, xi, target):
        # Update the weights
        forw_pass = self.activation(self.net_input(xi))
        error = (target - forw_pass)      # Here error is an error of single training example
        self.w_ += self.eta * error * xi
        self.b_ += self.eta * error
        return error**2

    
    def net_input(self, X):
    # Here, since we will use net_input(X) as input
    # output is basically a vector of shape [n_examples] (raw prediction)
        return np.dot(X, self.w_) + self.b_
    

    def activation(self, X):
    # Here, since we will use net_input(X) as input
    # X is basically a vector of shape [n_examples] (raw prediction)
        return X
    

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1.0, 0.0)
    

    def fit(self, X, y):

        """ 
        Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.
        
        Returns
        -------
        self : object
        """
        
        self._initialize_weights(X.shape[1])
        self.resid_ = []

        for i in range(self.n_iter):
                
            if self.shuffle: 
                X, y = self._shuffle(X, y)
            
            errors = [] # initialize the error array

            for xi, target in zip(X, y):      # Here we traverse through each training example

                error = self._update_weights(xi, target)
                np.append(errors, error)
            
            self.resid_.append( np.mean(errors) )
        
        return self
    
    
    def partial_fit(self, X, y):
        # Keep training the model (online training)
        # without reinitializing the weights

        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
            self.resid_ = []
        
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        
        return self