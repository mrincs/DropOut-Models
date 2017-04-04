# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 02:56:56 2016

@author: mrins
"""

"""
DropOut Paper

2 hidden layers, 4096 units each, ReLU Activation, max norm const

"""
import os
import six.moves.cPickle as pickle
import gzip

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict


from train import train

# For Debugging
#theano.config.optimizer='fast_compile'
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'warn'

n_epochs = 3

def ReLU(x):
    return T.maximum(0.0, x)
    
def Sigmoid(x):
    return T.nnet.sigmoid(x)
    
def Tanh(x):
    return T.tanh(x)
    
    
def gradientUpdates(cost, params, learningRate):
        
    assert learningRate > 0 and learningRate < 1
    
    updates = OrderedDict()           
    gparams = [T.grad(cost, param) for param in params]   
    for param, gparam in zip(params, gparams):
        updates[param] = param - learningRate * gparam
    return updates
    
    
def gradientUpdateUsingMomentum(cost, params, learningRate, momentum = 0.9):
    
    assert learningRate > 0 and learningRate < 1
    assert momentum > 0 and momentum < 1
    
    updates = OrderedDict()
    gparams = [T.grad(cost, param) for param in params]  
    for param, gparam in zip(params, gparams):
        velocity = theano.shared(param.get_value()*0, broadcastable=param.broadcastable)
        updates[velocity] = velocity * momentum + (1-momentum) * gparam
        updates[param] = param -  learningRate * velocity
        
    return updates
 
   
def gradientUpdatesWithMaxNormConstraint(cost, params, learningRate, maxNorm = 10, epsilon = 1e-7):
   
    assert learningRate > 0 and learningRate < 1
    
    updates = OrderedDict()
    gparams = [T.grad(cost, param) for param in params] 
    for param, gparam in zip(params, gparams):
        updates[param] = param - learningRate * gparam
        columnNormsOfUpdatedParams = T.sqrt(T.sum(T.sqr(updates[param]), axis=0))
        desiredColumnNormsOfUpdatedParams = T.clip(columnNormsOfUpdatedParams, 0, maxNorm)
        scale = (epsilon + desiredColumnNormsOfUpdatedParams) / (epsilon + columnNormsOfUpdatedParams)
        updates[param] = updates[param] * scale
        
    return updates   
    
    
class HiddenLayer(object):
    
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=None):
         
        self.input = input
        self.W, self.b = self.initializeParams(W, b, rng, n_in, n_out)
        self.params = [self.W, self.b]
                
        lin_output = T.dot(input, self.W) + self.b
        self.output = activation(lin_output)
        
    def initializeParams(self, W, b, rng, n_in, n_out):
        
        assert (n_in+n_out) != 0
        
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6/(n_in+n_out)),
                    high = np.sqrt(6/(n_in+n_out)),
                    size = (n_in, n_out)
                ),
                dtype = theano.config.floatX
            )
                
            W = theano.shared(value = W_values, name = "W", borrow = True)
        
        if b is None:
            b_values = np.zeros((n_out,), dtype = theano.config.floatX)
            b = theano.shared(value = b_values, name = "b", borrow = True)
            
        return [W, b]
        

def DropOutFromLayer(rng, layer, prob):
     """'prob' is probability of dropping an unit from 'layer'
        'mask' creates binary vector of zeros and ones to suppress any node 
     """
     srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
     mask = srng.binomial(n=1, p=1-prob, size=layer.shape)
     return layer * T.cast(mask, theano.config.floatX)
     
     
class HiddenLayerWithDropOut(HiddenLayer):
    
    def __init__(self, rng, input, n_in, n_out, dropOut, W=None, b=None, activation=None):
        super(HiddenLayerWithDropOut, self).__init__(
            rng = rng,
            input = input,
            n_in = n_in,
            n_out = n_out,
            W = W,
            b = b,
            activation = activation
            )
            
        self.output = DropOutFromLayer(rng, self.output, dropOut)
        
        
class LogisticRegression(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):

        self.W, self.b = self.initializeParams(W, b, n_in, n_out)
         
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
        
        self.params = [self.W, self.b]
        
        self.input = input
        
        
    def initializeParams(self, W, b, n_in, n_out):
        
        if W is None:
            W = theano.shared(
                value = np.zeros((n_in, n_out),
                                 dtype = theano.config.floatX),
                        name = "W",
                        borrow = True
                    )            
        if b is None:
            b = theano.shared(
                value = np.zeros((n_out,),
                                 dtype = theano.config.floatX),
                        name = "b",
                        borrow = True
                    )        
        return [W, b]
        
        
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        
        
    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
      

           
class MLPwithDropOut(object):
    
    def __init__(self, rng, input, n_in, n_hidden, dropOut, n_out, activation):
        
        self.n_hidden_layers = len(n_hidden)
        self.hiddenLayers = []
        
        for i in range(self.n_hidden_layers): 
            if i == 0:
                inputLayer = DropOutFromLayer(rng, input, dropOut[0])
                inputLayerSize = n_in             
            else:
                inputLayer = self.hiddenLayers[i-1].output
                inputLayerSize = n_hidden[i-1]
            
            hiddenLayer = HiddenLayerWithDropOut(
                rng = rng,
                input = inputLayer,
                n_in = inputLayerSize,
                n_out = n_hidden[i],
                dropOut = dropOut[i],
                activation = activation
            )
            self.hiddenLayers.append(hiddenLayer)
        
        self.outputLayer = LogisticRegression(
            input = self.hiddenLayers[-1].output,            
            n_in = n_hidden[-1],
            n_out = n_out
        )
        
        self.negative_log_likelihood = self.outputLayer.negative_log_likelihood
        
        self.errors = self.outputLayer.errors
        
        self.params = []
        for i in range(self.n_hidden_layers):
            self.params.extend(self.hiddenLayers[i].params)
        self.params.extend(self.outputLayer.params)
                
        self.input = input
        

        
    def getCostAndUpdates(self, y, learningRate = 0.01):
        
        assert y is not None
        
        cost = self.negative_log_likelihood(y)
        updates = gradientUpdatesWithMaxNormConstraint(cost, self.params, learningRate)
        
        return (cost, updates)
        
        
        
class DropOut_NN_ReLU(MLPwithDropOut):
    
    def __init__(self, rng, input):
        MLPwithDropOut.__init__(
            self,
            rng = rng,
            input = input,
            n_in = 28*28,
            n_hidden = [4096, 4096],
            dropOut = [0.1, 0.5, 0.5],
            n_out = 10,
            activation = ReLU
            )



def test_ANNs_onMNIST(learning_rate=0.01, L1_reg=0, L2_reg=0.001,
             dataset='mnist.pkl.gz', batch_size=20):
    
#    :type n_epochs: int
#    :param n_epochs: maximal number of epochs to run the optimizer
#
#    :type dataset: string
#    :param dataset: the path of the MNIST dataset file from
#                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

        
    datasets = load_data(dataset)
        
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
        
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y') 
    
                
    # Optimization Parameters
    learning_rate = 0.01
    momentum = 0.95
    
    
    ###########################
    # BUILD TRAINING MODELS #
    ###########################
        
    print('... Designing the model')
    
    rng = np.random.RandomState(7341)
    
    standard_NN_model_1 = DropOut_NN_ReLU(
            rng = rng,
            input = x,
            )
        
    model = standard_NN_model_1
    
    model.cost, model.updates = \
                model.getCostAndUpdates(y, learning_rate)
                
                
    

    
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=model.cost,
        updates=model.updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
#    print('.... Model Graph Round 1')    
#    theano.printing.debugprint(model.outputLayer.p_y_given_x)
    
    


    test_model = theano.function(
        inputs=[index],
        outputs=model.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=model.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

   

    ###############
    # TRAIN MODEL #
    ###############

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    
    numBatches = [n_train_batches, n_valid_batches, n_test_batches]
    
    print('... Training for round 1')
    train(numBatches, train_model, validate_model, test_model,
          numEpochs = n_epochs,
          validationFrequency = validation_frequency)


           
           
def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... Loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    
    
    print('Training Set:    ',len(train_set[0]))
    print('Validation Set:  ',len(valid_set[0]))
    print('Test Set:        ',len(test_set[0]))

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
    
    
    
if __name__ == '__main__':
    test_ANNs_onMNIST()