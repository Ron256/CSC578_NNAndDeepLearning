"""
Ronaldlee Ejalu
 
CSC 578 HW 5: Backprop HyperParamters

CSC 578 Fall 2020

NN578_network2.py
==============

Modified from the NNDL book code "network2.py".

network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
import json
import random
import sys
import numpy as np


#### Definitions of the cost functions (as function classes)
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output ``y``.
        """
        return 0.5*np.linalg.norm(y-a)**2

    ## nt: addition
    @staticmethod
    def derivative(a, y):
        """Return the first derivative of the function."""
        return -(y-a)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def derivative(a, y):
        """Return the first derivative of the function."""
        # Since we expect a to be between 0 and 1, if 
        # is either 0 or 1 just return 0.
        # this is equation 72 in NNDL (chapter 3)
        #print('shape of a: %s'%str(a.shape))
        #print('shape of y: %s'%str(y.shape))
        # to avoid ambiguity, use a.any() or a.all
        if a.all == 0 or a.all == 1:
            return 0
        else:
            return (a - y) / (a * (1-a)) # derivative of the CrossEntropy with respect to the output of the neuron.
            
class LogLikelihood(object):
    
    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with output a and desired output y
        The original function that Michael Nielsen talks about is 𝐶=(−1/𝑛) * ∑_x (ln𝑎^𝐿sub𝑦)
        """
        return np.nan_to_num(-np.log(a[np.argmax(y)]))[0]
    
    @staticmethod
    def derivative(a, y):
        # derive the number of elements in the array
        num = a.shape
        
        # initialize an array of zeros of the logs whose number of elements 
        # are derived with the help of the number of elements of the output a
        
        logL = np.zeros(num)
        
        # calculate the derivative of the loglikelihood when the desired/target output is 1
        targetInd = np.argmax(y)
        if a[targetInd][0] == 0:
            return logL
        else:
            logL[targetInd] = -1/a[targetInd]
            
        return logL # return the array with logs
        
        
    
#### Definitions of the activation functions (as function classes)
class Sigmoid(object):
    @staticmethod
    def fn(z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    @classmethod
    def derivative(cls,z):
        """Derivative of the sigmoid function."""
        return cls.fn(z)*(1-cls.fn(z))

class Softmax(object):
    @staticmethod
    # Parameter z is an array of shape (len(z), 1).
    def fn(z):
        """
        The softmax of vector z.
        From the slides of week4, this is expressed as: s_j^L=ⅇ^(z_j^L )/(∑_k▒ⅇ^(z_k^L ) )
        In normal words, it is the difference between each vector element 
        and the max divided by the sum of elements in the vector z.
        """
        
        # if z is large, e^z becomes large which causes an overflow
        # so, we shift z's by subtracting max(z) from all z's
        
        m_z = np.max(z)
        exp_z = np.exp(z - m_z)
        softMax_z = exp_z / exp_z.sum()
        
        return softMax_z
    @classmethod
    def derivative(cls,z):
        """Derivative of the softmax.  
        REMEMBER the derivative is an N*N matrix.
        """
        a = cls.fn(z) # obtain the softmax vector
        return np.diagflat(a) - np.dot(a, a.T)
    
class Tanh(object):
    @staticmethod
    def fn(z):
        """The tanh function.
        From the week4 slides, tanh(z) = (ⅇ^z - ⅇ^(-z))/(ⅇ^z  + ⅇ^(-z) )
        """
        return (np.exp(z) - np.exp(-z)) /  (np.exp(z) + np.exp(-z))

    @classmethod
    def derivative(cls,z):
        """Derivative of the tanh function.
        this is expressed as as 1 - tanh(z)^2
        """
        return (1 - (cls.fn(z) ** 2))
        
class ReLU(object):
    
    @staticmethod
    def fn(z):
        """
        relu = max(0, z) 
        we know R(z) = z if z > 0  and R(z)=0 if z <= 0        
        """
        
        z[z <= 0] = 0
        return z
        
    @classmethod
    def derivative(cls, z):
        """
        Derivative of ReLu
        R^' (z) = 1 if z > 0
        R^' (z) = 0 if z < 0
        """
        r_z = cls.fn(z)
        
        # if r_z is is greater than 0 then 1 else 0
        return 1 * (r_z > 0)
        
    
#### Main Network class
class Network(object):

    ## Additional keyword arguments for hyper-parameters
    def __init__(self, sizes, cost=CrossEntropyCost, act_hidden=Sigmoid,
                 act_output=None, regularization=None, lmbda=0.0,
                 dropoutpercent=0.0):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that method).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        
        self.set_parameters(cost, act_hidden, act_output, regularization, lmbda,
                            dropoutpercent)

    ## THIS NEEDS CHANGES NEEDED.
    ## nt: convenience function for setting network hyperparameters
    def set_parameters(self, cost=QuadraticCost, act_hidden=Sigmoid,
                       act_output=None, regularization=None, lmbda=0.0,
                       dropoutpercent=0.0):
        self.cost=cost
        self.act_hidden = act_hidden
        if act_output == None:
            self.act_output = self.act_hidden
        else:
            self.act_output = act_output
        
        # if cost function was set to anything besides QuadracticCost when act_output is equal to tanh set the cost to the QuadracticCost
        if self.act_output == Tanh and self.cost != QuadraticCost:
            # printing warning
            print("Tanh only accepts QuadraticCost cost function. Changing to QuadracticCost.")
            self.cost = QuadraticCost
        
        # LogLikelihood cost function should only be used when act_output is equal for Softmax
        if self.act_output == Softmax and self.cost != LogLikelihood:
            self.cost = LogLikelihood
        
        self.regularization = regularization
        self.lmbda = lmbda
        self.dropoutpercent = dropoutpercent
        
    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    ## CHANGES NEEDED
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        
        
        for b, w in zip(self.biases, self.weights):
            ## THIS NEEDS (FURTHER) CHANGE. 
            ##  The original code (commented) doesn't work any more.  
            ##  The new scheme is written which utilizes a function class.  
            ##  But this is still incorrect because the activation of 
            ##  the output layer has is not considered (which needs to be).
            a = sigmoid(np.dot(w, a)+b)
            
            
            """    
            if self.dropoutpercent > 0.0 and self.dropoutpercent < 1.0:
                dropoutMask_L = []
                    
                # ration of nodes to retain
                p = 1 - self.dropoutpercent
            
                # loop through the different layers to access the different neurons to 
                # generate a scaled dropout mask list which will be assigned to the dropoutmask
                for num_l in range(1, len(self.sizes)):
                    # generate a dropout mask for each layer
                    # the numerator is is going to be a boolean array with 1s and 0s 
                    # and we scale the values by diving by the percentage of nodes to be retained.
                    u_mask = np.random.binomial(1, p, size=(self.sizes[num_l], 1)) / p
                    # print('%s iteration: umask is %s' %(str(num), str(u_mask)))
                
                    # append the mask on the list
                    dropoutMask_L.append(u_mask)
                    #print('contents of dropoutMask_L: \n %s' %dropoutMask_L)
                    # assign the generated dropoutmask_L (list) to dropoutmask object
                a *= dropoutMask_L[num - 1]
            """
            
            
        return a
    
    ## nt: additional parameter 'no_convert' to control the vectorization of the target.
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            no_convert=True):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        ## nt: additional lines to possibly change the dataset
        ##   in case the output layer's activation function is tanh.
        if self.act_output == Tanh:
            training_data = tanh_data_transform(training_data)
            if evaluation_data is not None:
                evaluation_data = tanh_data_transform(evaluation_data)
        
        ## nt: back to the original code..
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            #random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # print('QA testing: %s: %s: %s: %s'%(str(mini_batch), str(eta), str(lmbda), str(len(training_data))))
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            
            ## nt: from here, most lines are for printing purpose only.
            print ("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda) # nt: for cost, always NO convert (default) for training
                training_cost.append(cost)
                print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True) # nt: for accuracy, always _DO_ convert (argmax) for training
                training_accuracy.append(accuracy)
                print ("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                ## nt: changed the last parameter convert
                if no_convert:
                    cost = self.total_cost(evaluation_data, lmbda) # nt: if test/val data is already vectorized for y
                else:
                    cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print ("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                ## nt: changed the last parameter convert
                if no_convert:
                    accuracy = self.accuracy(evaluation_data, convert=True) #nt: _DO_ convert (argmax)
                else:
                    accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print ("Accuracy on evaluation data: {} / {}".format(
                    ## nt: This seems like a bug!
                    #self.accuracy(evaluation_data), n_data))
                    accuracy, n_data))
            print ('')
        return training_cost, training_accuracy , \
                evaluation_cost, evaluation_accuracy
            
    ##  CHANGES NEEDED. 
    ##  This original code is hard-coding L2 norm.  You need to change
    ##  so that the parameter self.regularization is used and do the
    ##  appropriate regularization.
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        # L1 regularization
        if self.regularization == 'L1':
            # From the Differential of the cost:
            # ∂c/∂ω=∂c/∂ω+λ/n  sgn⁡(ω)
            
            # we need to determine the different values of the weights 
            # 1 if w > 0
            # 0 if w == 0
            # -1 if w < 0
            
            # define a list where to store the values of the weights
            sgn_Lw = self.derive_Sgnw(self.weights)

            #    sgn_Lw.append(ww)
            #print('sgn_Lw %s' %sgn_Lw)
            #sgn_Lw = np.asarray(sgn_Lw)
            #print('sgn_Lw : %s' %sgn_Lw.shape) 
            
            # Apply the weight update rule
            # ω-(nλ/n)  sgn⁡(w)-n (∂C_0)/∂w
            
            self.weights = [w - (eta*lmbda/n)*sgn_w - (eta/len(mini_batch))*nw 
                                for w, nw, sgn_w in zip(self.weights, nabla_w, sgn_Lw)]
                                
            # L2 regularization is the default regularization
        else: 
            self.weights = [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw 
                            for w, nw in zip(self.weights, nabla_w)]
                            
            self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        

    ## CHANGES NEEDED.
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        
        actL_count = 1 # initialized to track the number of activation layers
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            
            if actL_count == len(self.biases):
                
                ## nt: changed to use function class for
                ##   the activation function of hidden layer(s).
                #activation = sigmoid(z)
                #activation = (self.act_hidden).fn(z)
            
                # if activation output is None
                if self.act_output == None:
                    activation = (self.act_hidden).fn(z)
                else:
                    activation = (self.act_output).fn(z)
                # activation = (self.act_output).fn(z)
                
                
            else:
                activation = (self.act_hidden).fn(z)
                
                """
                
                # dropout applied after activation
                if self.dropoutpercent > 0.0 and self.dropoutpercent < 1.0:
                    print('contents of dropout mask %s' %self.dropoutmask_u1)
                    # applying the mask to the activations of the hidden layer
                    activation *= self.dropoutmask_u1[actL_count - 1]
                    
                """
                
                if self.dropoutpercent > 0.0 and self.dropoutpercent < 1.0:
                    dropoutMask_L = []
                    
                    # ration of nodes to retain
                    p = 1 - self.dropoutpercent
            
                    # loop through the different layers to access the different neurons to 
                    # generate a scaled dropout mask list.
                    #for num in range(num_layers):
                    for num in range(1, len(self.sizes) - 1):
                        # generate a dropout mask for each layer
                        # the numerator is is going to be a boolean array with 1s and 0s 
                        # and we scale the values by diving by the percentage of nodes to be retained.
                        u_mask = np.random.binomial(1, p, size=(self.sizes[num], 1)) / p
                        # print('%s iteration: umask is %s' %(str(num), str(u_mask)))
                
                        # append the mask on the list
                        dropoutMask_L.append(u_mask)
                    #print('contents of dropoutMask_L: \n %s' %dropoutMask_L)
                    # assign the generated dropoutmask_L (list) to dropoutmask object
                    activation *= dropoutMask_L[actL_count - 1]
                    
            activations.append(activation)
            actL_count += 1

        # backward pass
        ## nt: Cost and activation functions are parameterized now.
        ##     Call the activation function of the output layer with z.
        #delta = (self.cost).delta(zs[-1], activations[-1], y)
        a_prime = (self.act_output).derivative(zs[-1]) # nt: da/dz
        c_prime = (self.cost).derivative(activations[-1], y) # nt: dC/da
        
        # nt: compute delta -- separate case for Softmax
        if self.act_output == Softmax:
            delta = np.dot(a_prime, c_prime) 
        else:
            delta = c_prime * a_prime # nt: dC/da * da/dz

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
            ## nt: Changed to call the activation function of the 
            ##  hidden layer with z.
            #sp = sigmoid_prime(z)
            sp = (self.act_hidden).derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    
    ## CHANGES NEEDED. 
    ##  This original code is hard-coding L2 norm.  You need to change
    ##  so that the parameter self.regularization is used and do the
    ##  appropriate regularization.
    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        
        # L2 regularization
        if self.regularization == 'L2':
            ## C = 
            cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
            
        # L1 regularization
        elif self.regularization == 'L1':
            ## C = 
            cost += (lmbda/len(data))* \
                            np.sum([np.sum(np.abs(w)) for w in self.weights])
                            
        # default: when no regularization is specified in the SGD()
        else:
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        
    def derive_Sgnw(self,w):
        """
        function that returns sgn(w)
        ** needs to be revisted
        """
        for weig in w:
            # for each layer there are multiple elements
            for l in range(len(weig)):
                ww = weig[l]
                
                # if w > 0 assign 1 to the weight or else 0
                ww[ww > 0] = 1
                
                # if w < 0 then assign -1 to the weight
                ww[ww < 0] = -1
        
        return w
            

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

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
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

## nt: new function to generate a new dataset compatible with tanh.
## 6/2019: updated with copy.deepcopy().  This ensures the right solution.
import copy
def tanh_data_transform(dataset):
    xlist = [x for (x,y) in dataset]
    ylist = copy.deepcopy([y for (x,y) in dataset])
    for lst in ylist:
        lst[lst == 0] = -1 # replace 0's by -1's
    return list(zip(xlist, ylist))
