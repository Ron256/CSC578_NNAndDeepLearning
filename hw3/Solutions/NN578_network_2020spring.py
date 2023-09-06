"""
My Solution Code (as of 4/17/2020)

NN578_network_2020spring.py
==============

Modified from the NNDL book code "network.py" to be 
compatible with Python 3.  
Also from "network2.py", the function ("load(filename)") is
added to this file, renamed as "load_network(filename)".
This function loads a saved network encoded in a json file.

"""

"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import json

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, stopaccuracy=1.0):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        # 9/2018 nt: Addition
        trn_results = []
        tst_results = []
        
        for j in range(epochs): #xrange(epochs):
            #random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                
            ###9/2018 nt: my modification
            # At the end of every epoch,
            # First call evaluate on the training data, always
            train_results = self.evaluate(training_data)
            print ("[Epoch {0}] Training: MSE={1:.8f}, CE={2:.8f}, LL={3:.8f}, Correct: {4}/{5}, Acc: {6:.8f}".format( 
                   j, train_results[2], train_results[3], train_results[4], train_results[0], n, train_results[1]))
            trn_results.append(train_results)
            
            # And if test_data is passed, evaluate the test data and print the results
            if test_data:
                # 9/2018 nt: Added to expand the target (y) in test_data to a one-hot-vector notation
                # 4/17/2020
                #             
                test_results = self.evaluate(test_data)
                
                print ("          Test:     MSE={1:.8f}, CE={2:.8f}, LL={3:.8f}, Correct: {4}/{5}, Acc: {6:.8f}".format( 
                       j, test_results[2], test_results[3], test_results[4], test_results[0], n_test, test_results[1]))
            else:
                test_results = []
            tst_results.append(test_results)
            
            ## 6/2019 nt: Early stopping -- comparing against stopaccuracy
            if train_results[1] >= stopaccuracy:
                break
            
        # After all epochs are run, return the two result lists in a list
        return [trn_results, tst_results]
        

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        ## 9/2018 nt: change to static allocation
        activations = [np.zeros((y, 1)) for y in self.sizes] #[x] # list to store all the activations, layer by layer
        activations[0] = x # assign the input layer
        
        zs = [] # list to store all the z vectors, layer by layer
        aindex = 1
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            #activations.append(activation)
            ## 9/2018 nt: change
            activations[aindex] = activation
            aindex += 1
            
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        ### 9/2018 nt: my modification
        MSEtotal = 0.0
        CEtotal = 0.0
        LHtotal = 0.0
        correctcount = 0
        for (x, y) in test_data:
            a = self.feedforward(x)
            
            MSEtotal += 0.5*np.linalg.norm(a-y)**2
            CEtotal += np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
            ## 4/16/2019: 
            # Numpy array subscript returns an array with one element rather than a scalar.
            # One way to get to the scalar is to call item() -- BTW which returns type 
            #   <class 'numpy.float64'>
            #LHtotal += np.nan_to_num(-np.log((a[np.argmax(y)]).item())) 
            # or np.nan_to_num(-np.log(a[np.argmax(y)]))[0] 
            
            # 4/21/2019: Further modification to maximum likelihood
            LHtotal += np.sum(np.nan_to_num(-y*np.log(a)))
            
            # 4/17/2020 Got rid of the case for a scalar y
            targetIndex = np.argmax(y)

            if np.argmax(a) == targetIndex:
                correctcount += 1
        #
        n = len(test_data)
        return [correctcount, correctcount/n, MSEtotal/n, CEtotal/n, LHtotal/n]
        

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#### Savinging a Network to a json file
def save_network(net, filename):
    """Save the neural network to the file ``filename``."""
    data = {"sizes": net.sizes,
            "weights": [w.tolist() for w in net.weights],
            "biases": [b.tolist() for b in net.biases]#,
            #"cost": str(net.cost.__name__)
           }
    f = open(filename, "w")
    json.dump(data, f)
    f.close()
        
#### Loading a Network from a json file
def load_network(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    #cost = getattr(sys.modules[__name__], data["cost"])
    #net = Network(data["sizes"], cost=cost)
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorize_target(n, target):
    """Return an array of shape (n,1) with a 1.0 in the target position
    and zeroes elsewhere.  The parameter target is assumed to be
    an array of size 1, and the 0th item is the target position (1).

    """
    e = np.zeros((n, 1))
    e[int(target[0])] = 1.0
    return e