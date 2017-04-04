# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:45:11 2016

@author: mrins
"""

import os
import sys
import timeit
import numpy as np


'''Simple training module operated on mini batches. 
 It goes through the batches for each training set and periodically validates the model.
 If validation of model is better than previous performances, it operates on test data
'''
def train(numBatches, train_model, validate_model, test_model,
          numEpochs = 10,
          validationFrequency = 0.1):
    
    best_validation_loss = np.inf
    assert len(numBatches) == 3
    numTrainBatches, numValidateBatches, numTestBatches = numBatches[0], numBatches[1], numBatches[2]
    
    start_time = timeit.default_timer()
    
    for epoch in range(numEpochs):
                
        for minibatchIdx in range(numTrainBatches):
            
            train_loss = train_model(minibatchIdx)
            
            iterationIdx = epoch * numTrainBatches + minibatchIdx
                        
            if ((iterationIdx+1) % validationFrequency) == 0:
                
                validation_losses = [validate_model(idx) for idx in range(numValidateBatches)]
                this_validation_loss = np.mean(validation_losses)
                
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatchIdx + 1,
                        numTrainBatches,
                        this_validation_loss * 100.
                    )
                )                
                
                
                if this_validation_loss < best_validation_loss:
                    
                    best_validation_loss = this_validation_loss
                    best_iter = iterationIdx
                    test_losses = [test_model(idx) for idx in range(numTestBatches)]
                    test_loss = np.mean(test_losses)
                    
                    print(
                        '     epoch %i, minibatch %i/%i, test error of best model %f %%' %
                        (
                            epoch,
                            minibatchIdx + 1,
                            numTrainBatches,
                            test_loss * 100.
                        )
                    )

    print_execution_details(best_validation_loss, best_iter, test_loss, start_time)                
    return
        
        
        
'''
It modifies upon train module by including an extra parameter called patience
'''       
def train_v2(numBatches, train_model, validate_model, test_model, 
             numEpochs = 10, 
             patience = 10000,  # look as this many examples regardless
             patience_increase = 2,  # wait this much longer when a new best is found
             improvement_threshold = 0.995  # a relative improvement of this much is considered significant
             ):
    
    assert len(numBatches) == 3
    numTrainBatches, numValidateBatches, numTestBatches = numBatches[0], numBatches[1], numBatches[2]
    best_validation_loss = np.inf
    validationFrequency = min(numTrainBatches, patience // 2)
    
    start_time = timeit.default_timer()
    
    for epoch in range(numEpochs):
        
        for minibatchIdx in range(numTrainBatches):
            
            train_loss = train_model(minibatchIdx)
            
            iterationIdx = (epoch-1) * numBatches + minibatchIdx
            
            if ((iterationIdx+1) % validationFrequency) == 0:
                
                validation_losses = [validate_model(idx) for idx in range(numValidateBatches)]
                this_validation_loss = np.mean(validation_losses)
                
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatchIdx + 1,
                        numTrainBatches,
                        this_validation_loss * 100.
                    )
                )                
                
                
                if this_validation_loss < best_validation_loss:
                    
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iterationIdx * patience_increase)                        
                    
                    best_validation_loss = this_validation_loss
                    best_iter = iterationIdx
                    test_loss = [test_model(idx) for idx in range(numTestBatches)]
                    
                    print(
                        '     epoch %i, minibatch %i/%i, test error of best model %f %%' %
                        (
                            epoch,
                            minibatchIdx + 1,
                            numTrainBatches,
                            np.mean(test_loss ) * 100.
                        )
                    )
                    
                    if patience <= iter:
                        print('Finishing before proposed number of epochs')
                        print_execution_details(best_validation_loss, best_iter, test_loss, start_time)
                        return
                        
    print_execution_details(best_validation_loss, best_iter, test_loss, start_time)            
    return
    
    
        
def print_execution_details(best_validation_loss, best_iter, test_loss, start_time):
    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_loss * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    return
    
    

'''Assuming hyperparameters are already tuned, theres no separate validation set
Train and validation set are merged to train model on more data
'''
def train_v3(numBatches, train_model, test_model,
          numEpochs = 10):
    
    assert len(numBatches) == 2
    numTrainBatches, numTestBatches = numBatches[0], numBatches[1]
    
    start_time = timeit.default_timer()
    
    for epoch in range(numEpochs):
        
        train_losses = [train_model(minibatchIdx) for minibatchIdx in range(numTrainBatches)]
        print(train_losses)
        train_loss = np.mean(train_losses)
        print(
            'epoch %i, train error %f %%' %
                (
                    epoch,
                    train_loss * 100.
                )
            )
            
        test_losses = [test_model(minibatchIdx) for minibatchIdx in range(numTestBatches)]
        test_loss = np.mean(test_losses)
        print(test_losses)
        print(
            'epoch %i, test error %f %%' %
                (
                    epoch,
                    test_loss * 100.
                )
            )

    end_time = timeit.default_timer()

    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    return
        
        
        
def shuffle_data(X, y):
    ''' Shuffles the original data
    
    X : Input Vectors (Numpy matrix)
    y : Target Variables (Numpy array)
    
    '''
    len_X = len(X)
    assert len_X == len(y)
    
    shuffled_indices = np.arange(len_X)
    np.random.shuffle(shuffled_indices)
    
    shuffled_X = np.zeros(len_X)
    shuffled_y = np.zeros(len_X)
    
    for idx in range(len_X):
        shuffled_X[idx] = X[shuffled_indices[idx]]
        shuffled_y[idx] = y[shuffled_indices[idx]]
        
    return shuffled_X, shuffled_y
    