# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:56:55 2017

@author: mrins
"""

"""
DropOut Paper

3 hidden layers, 1024 units each, ReLU Activation, max norm const

"""
import os
import six.moves.cPickle as pickle
import gzip

import numpy as np
import theano
import theano.tensor as T

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
           
    gparams = [T.grad(cost, param) for param in params]   
    updates = [
        (param, param - learningRate * gparam)
        for param, gparam in zip(params, gparams)
        ]
    return updates
    
    
def gradientUpdateUsingMomentum(cost, params, learningRate, momentum):
    
    assert learningRate > 0 and learningRate < 1
    assert momentum > 0 and momentum < 1
    
    updates = []
    for param in params:
        paramUpdate = theano.shared(param.get_value()*0, broadcastable=param.broadcastable)
        updates.append((param, param - learningRate * paramUpdate))
        updates.append((paramUpdate, momentum * paramUpdate + (1 - momentum) * T.grad(cost, param)))
    return updates
    
    
    
    
class HiddenLayer(object):
    
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=None):
         
        assert activation == ReLU
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
        

def SkipHiddenUnits(rng, layer, dropOut_prob):
     """ Drop out units from layers during training phase
        'mask' creates binary vector of zeros and ones to suppress any node 
     """
     retain_prob = 1 - dropOut_prob
     srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
     mask = srng.binomial(n=1, p=retain_prob, size=layer.shape, dtype=theano.config.floatX)
     return T.cast(layer * mask, theano.config.floatX)
     
     
def ScaleWeight(layer, dropOut_prob):
    """ Rescale weights for averaging of layers during validation/test phase
    """
    retain_prob = 1 - dropOut_prob
    return T.cast(retain_prob * layer, theano.config.floatX)
    
    
    
def SelectiveDropOut(rng, layer, dropOut_prob):
     """'dropOut_prob' is probability of dropping an unit from 'layer' individually.
         'dropOut_prob' is an array here.
     """
     layer_size = len(dropOut_prob)
     assert layer_size == layer.shape[0]
     retain_prob = np.ones(layer_size) - dropOut_prob
     mask = np.zeros(layer_size)
     srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
     
     for i in range(layer_size):
         mask[i] = srng.binomial(n=1, p=retain_prob[i], size=1)
         
     return layer * T.cast(mask, theano.config.floatX)

     
     
class HiddenLayerWithDropOut(HiddenLayer):
    
    def __init__(self, rng, input, n_in, n_out, dropOut, train_phase=1, W=None, b=None, activation=None):
        super(HiddenLayerWithDropOut, self).__init__(
            rng = rng,
            input = input,
            n_in = n_in,
            n_out = n_out,
            W = W,
            b = b,
            activation = activation
            )
        
        training_output = SkipHiddenUnits(rng, self.output, dropOut)
        predict_output = ScaleWeight(self.output, dropOut)
        self.output = T.switch(T.neq(train_phase, 0), training_output, predict_output)
        
        
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
    
    def __init__(self, rng, input, n_in, n_hidden, dropOut, n_out, train_phase, activation):
        
        self.n_hidden_layers = len(n_hidden)
        self.hiddenLayers = []
        
        for i in range(self.n_hidden_layers): 
            if i == 0:
                inputLayer = SkipHiddenUnits(rng, input, dropOut[0])
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
                train_phase = train_phase,
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
        updates = gradientUpdateUsingMomentum(cost, self.params, learningRate, momentum = 0.95)
        
        return (cost, updates)
        
        
        
class DropOut_NN_ReLU(MLPwithDropOut):
    
    def __init__(self, rng, input, train_phase):
        MLPwithDropOut.__init__(
            self,
            rng = rng,
            input = input,
            n_in = 28*28,
            n_hidden = [10],
            dropOut = [0.1, 0.5],
            n_out = 10,
            train_phase = train_phase,
            activation = ReLU
            )



def test_ANNs_onMNIST(learning_rate=0.01, L1_reg=0, L2_reg=0.001,
                      dataset='mnist.pkl.gz', batch_size=20):
        
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
    train_phase = T.iscalar('train_phase')
    
                
    # Optimization Parameters
    learning_rate = 0.01
    momentum = 0.95
    
    
    ###########################
    # BUILD TRAINING MODELS #
    ###########################
        
    print('... Designing the model')
    
    rng = np.random.RandomState(7341)
            
    model = DropOut_NN_ReLU(rng = rng,
                            input = x,
                            train_phase = train_phase
                            )
    
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
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            train_phase: np.cast['int32'](1)
        }
    )
    
    print('.... Model Graph Round 1')    
    theano.printing.debugprint(model.outputLayer.p_y_given_x)
    
    


    test_model = theano.function(
        inputs=[index],
        outputs=model.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size],
            train_phase: np.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=model.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size],
            train_phase: np.cast['int32'](0)
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

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')


    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
            
    print('Training Set:    ',train_set_x.get_value().shape)
    print('Validation Set:  ',valid_set_x.get_value().shape)
    print('Test Set:        ',test_set_x.get_value().shape)
    
    return rval
    
    
    
if __name__ == '__main__':
    test_ANNs_onMNIST()