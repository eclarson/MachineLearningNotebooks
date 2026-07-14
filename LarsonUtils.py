import numpy as np
from scipy.special import expit
from numpy.linalg import pinv

# Note: this code uses several optimizations compared to the code in my notebooks. This should be slightly faster. 

class BinaryLogisticRegression:
    def __init__(self, eta, iterations=20, C=0.001):
        self.eta = eta
        self.iters = iterations
        self.C = C
        
    def __str__(self):
        if(hasattr(self,'w_')):
            return 'Binary Logistic Regression Object with coefficients:\n'+ str(self.w_) # is we have trained the object
        else:
            return 'Untrained Binary Logistic Regression Object'
        
    # convenience, private:
    @staticmethod
    def _add_bias(X):
        return np.hstack((np.ones((X.shape[0],1)),X)) # add bias term
    
    @staticmethod
    def _sigmoid(theta):
        return expit(theta) #1/(1+np.exp(-theta))
    
    # vectorized gradient calculation with regularization using L2 Norm
    def _get_gradient(self,X,y,g):
        gradient = X.T @ (y-g) / X.shape[0]
        gradient[1:] += -2 * self.w_[1:] * self.C
        
        return gradient

    def _get_direction(self,X,y):
        g = self.predict_proba(X, add_bias=False).ravel()
        reg = 2 * self.C * np.eye(X.shape[1])
        reg[0, 0] = 0      # don't regularize bias
        
        s = g * (1 - g)
        hessian = X.T @ (X * s[:, None]) + reg # calculate the hessian
        
        gradient = self._get_gradient(X,y,g)
        
        return np.linalg.solve(hessian, gradient)
    
    # public:
    def predict_proba(self,X,add_bias=True):
        # add bias term if requested
        Xb = self._add_bias(X) if add_bias else X
        return self._sigmoid(Xb @ self.w_) # return the probability y=1
    
    def predict(self,X):
        return (self.predict_proba(X)>0.5) #return the actual prediction
    
    
    def fit(self, X, y):
        Xb = self._add_bias(X) # add bias term
        num_samples, num_features = Xb.shape
        
        self.w_ = np.zeros(num_features) # init weight vector to zeros
        
        # for as many as the max iterations
        for _ in range(self.iters):
            gradient = self._get_direction(Xb,y)
            self.w_ += gradient*self.eta # multiply by learning rate 

            if np.linalg.norm(gradient) < 1e-6:
                break # if gradient is not large, just stop

class LogisticRegression:
    def __init__(self, eta, iterations=20, 
                 C=0.0001, ):
        self.eta = eta
        self.iters = iterations
        self.C = C
        self.classifiers_ = []
        # internally we will store the weights as self.w_ to keep with sklearn conventions
    
    def __str__(self):
        if(hasattr(self,'w_')):
            return 'MultiClass Logistic Regression Object with coefficients:\n'+ str(self.w_) # is we have trained the object
        else:
            return 'Untrained MultiClass Logistic Regression Object'
        
    def fit(self,X,y):
        num_samples, num_features = X.shape
        self.unique_ = np.sort(np.unique(y)) # get each unique class value
        num_unique_classes = len(self.unique_)
        self.classifiers_ = []
        for i,yval in enumerate(self.unique_): # for each unique value
            y_binary = np.array(y==yval).astype(int) # create a binary problem
            # train the binary classifier for this class
            
            hblr = BinaryLogisticRegression(eta=self.eta, iterations=self.iters, C=self.C)
            hblr.fit(X,y_binary)

            # add the trained classifier to the list
            self.classifiers_.append(hblr)
            
        # save all the weights into one matrix, separate column for each class
        self.w_ = np.hstack([x.w_ for x in self.classifiers_]).T
        
    def predict_proba(self,X):        
        return np.column_stack(
                [clf.predict_proba(X) for clf in self.classifiers_]
            )
    
    def predict(self,X):
        return self.unique_[np.argmax(self.predict_proba(X),axis=1)] # take argmax along row

#============================
# START: NEURAL NETWORK CODE
#============================
class Parameter:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

# generic class for creating layers
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.name = None

    def backward(self, sensitivity_in): pass
    def parameters(self): return []
    def set_parameters(self, parameters): pass 
    def zero_grad(self): pass

# generic class for creating loss functions
class Loss:
    def __init__(self): 
        self.y_true = None
        self.y_pred = None
        self.output = None
    def __call__(self, y_true, y_pred): pass
    def backward(self): pass 

# class for calling forward and backward of various layers, adding operations
class Model:
    def __init__(self, layers):
        self.layers = layers
        self._setup() 
        
    def __call__(self, input, sparse=False):
        output = input
        for layer in self.layers:
            output = layer(output)

        if sparse: return np.argmax(output, axis=0)  
            
        return output

    def __str__(self):
        tmp = f'Model: {len(self.layers)} operations.\n\n'
        for layer in self.layers:
            num = sum([p.value.size for p in layer.parameters()])
            tmp += f'{layer.name} \t | {num} Trainable parameters\n'

        return tmp

    def _setup(self):
        # need unique names for setting and getting
        for l in range(len(self.layers)):
            if self.layers[l].name is None:
                self.layers[l].name = f'{type(self.layers[l]).__name__}_{l}'
        
        # get complete list of all the parameters
        self.parameters = []
        for layer in self.layers:
            self.parameters.extend(layer.parameters())

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def add(self, operations):
        self.layers.extend(operations)
        self._setup() 
        
class MSELoss(Loss):
    def __call__(self, y_true, y_pred):
        self.y_diff = y_pred - y_true
        self.batch_size = y_true.shape[1]
        self.output = np.mean(np.power(self.y_diff, 2))
        return self.output

    def backward(self):
        return 2 * (self.y_diff) / self.batch_size

class Sigmoid(Layer):
        
    def __call__(self,input):
        self.input = input
        self.output = expit(input)
        return self.output

    def backward(self, sensitivity_in):
        s = self.output
        return s * (1 - s) * sensitivity_in

class Dense(Layer):
    def __init__(self, input_size, output_size):
        """Initialize weights with small random numbers."""
        num_elems = output_size*input_size
        tmp_weights = np.random.uniform(-1.0, 1.0, size=num_elems)
        
        self.weights = Parameter(tmp_weights.reshape((output_size, input_size)))
        self.bias = Parameter(np.zeros((output_size, 1)))
        
        super().__init__()

    def __call__(self, input):
        self.input = input
        self.output = self.weights.value @ self.input + self.bias.value
        return self.output

    def backward(self, sensitivity_in):
        self.weights.grad += sensitivity_in @ self.input.T #/ self.input.shape[1]
        self.bias.grad += np.sum(sensitivity_in, axis=1, keepdims=True) 
        
        return self.weights.value.T @ sensitivity_in

    def zero_grad(self):
        # ensure that numpy keeps original reference
        self.weights.grad.fill(0)
        self.bias.grad.fill(0)

    def parameters(self):
        # these are passing the numpy references back, so we can change outside of class
        return [self.weights,self.bias]

class Optimizer:
    def __init__(self, parameters): 
        self.parameters = parameters # passed by numpy reference
    def __str__(self):
        vals = sum([p.value.size for p in self.parameters])
        return f'Optimizer for {len(self.parameters)} layers with {vals} trainable parameters.'
    def step(self): pass
    def scheduler_step(self): pass


class SimpleSGD(Optimizer):
    def __init__(self, parameters, learning_rate):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        
    def step(self):
        # create our own optimizer, SGD for now
        # this references the original model parameters
        for param in self.parameters:
            param.value -= param.grad * self.learning_rate


class SGD(Optimizer):
    def __init__(self, parameters, learning_rate, momentum, decay_rate=0.9, step_size=10, min_learning_rate = 1e-6):
        self.parameters = parameters # passed by numpy reference
        
        self.learning_rate = learning_rate
        self.learning_rate_min = min_learning_rate

        self.decay_rate = decay_rate 
        self.step_size = step_size
        
        self.momentum = momentum # alpha value
        
        # for momentum, we need to cache the previous gradient
        self._cache = [np.zeros_like(p.grad) for p in parameters] 

    def step(self):
        # create our own optimizer, using momentum
        for param, cache in zip(self.parameters, self._cache):
            cache[:] = param.grad + self.momentum * cache
            param.value -= self.learning_rate * cache
            
    def scheduler_step(self, epoch):
        # just update the scheduler here
        if epoch !=0 and (epoch % self.step_size) == 0:
            self.learning_rate *= self.decay_rate
            self.learning_rate = max(self.learning_rate, self.learning_rate_min)

