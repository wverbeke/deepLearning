from keras import models
from keras import layers
from keras import optimizers 
from keras import callbacks

from treeToArray import *
from diagnosticPlotting import *

from ROOT import TFile
import numpy as np

from keras import backend as K


def tensorFlowSetNumThreads( num_threads ):
    K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads = num_threads, inter_op_parallelism_threads = num_threads ) ) )


#make name for model depending on its hyperparameters 
def denseModelName(num_hidden_layers, units_per_layer, activation, learning_rate, dropoutFirst, dropoutAll):
    model_name = 'model_{0}hiddenLayers_{1}unitsPerLayer_{2}_learningRate{3}'.format(num_hidden_layers, units_per_layer, activation, learning_rate)  
    model_name = model_name.replace('.', 'p')
    model_name += ( '_dropoutFirst' if dropoutFirst else '' )
    model_name += ( '_dropoutAll' if dropoutAll else '' )
    return model_name


def trainDenseClassificationModel(train_data, train_labels, validation_data, validatation_labels, train_weights = None, validation_weights = None, num_hidden_layers = 5, units_per_layer = 256, activation = 'relu', learning_rate = 0.0001, dropoutFirst=True, dropoutAll=False, dropoutRate = 0.5, num_epochs = 20, num_threads = 1):

    model = models.Sequential()

    #assume 2D input array
    input_shape = ( len(train_data[-1]), )

    #start with normalization layer to deal with potentially unprocessed data 
    model.add( layers.BatchNormalization( input_shape = input_shape ) )
    
    #add hidden densely connected layers 
    for x in range(num_hidden_layers) :
        model.add( layers.Dense( units_per_layer, activation = activation ) )
        if (dropoutFirst and x == 0) or dropoutAll:
            model.add( layers.Dropout( dropoutRate ) )
    
    #output layer
    model.add( layers.Dense(1, activation = 'sigmoid') )

    #print out model configuration
    model.summary()

    #set model objectives
    model.compile(
        optimizer = optimizers.RMSprop(lr=0.0001),
        loss = 'binary_crossentropy',
        metrics = ['accuracy',]
    )

    #name of file in which model will be saved
    model_output_name = denseModelName( num_hidden_layers, units_per_layer, activation, learning_rate, dropoutFirst, dropoutAll)

    #cut off training at convergence and save model with best validation
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor = 'acc',
            patience = 4
        ),
        callbacks.ModelCheckpoint(
            monitor = 'val_acc',
            filepath = model_output_name + '.h5',
            save_best_only = True
        )
    ]

    #set number of threads to use in training        
    tensorFlowSetNumThreads( num_threads ) 

    #train model
    training_history = model.fit(
        train_data,
        train_labels,
        sample_weight = (None if train_weights.size == 0 else train_weights),
        epochs = num_epochs,
        batch_size = 128,
        validation_data = (validation_data, validatation_labels, (None if validation_weights.size == 0 else validation_weights) ),
        callbacks = callbacks_list
    )

    #plot loss and accuracy as a function of epochs
    plotAccuracyComparison( training_history, model_output_name )
    plotLossComparison( training_history, model_output_name )

    #output name of model so that caller can access the correct file and load the best (saved) model
    return model_output_name


   
if __name__ == '__main__':
    pass

