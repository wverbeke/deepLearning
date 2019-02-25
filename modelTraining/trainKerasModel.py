from keras import models
from keras import layers
from keras import optimizers 
from keras import callbacks
from keras import backend as K
import h5py
import tensorflow as tf

#import other parts of framework
from treeToArray import *
from diagnosticPlotting import *


def tensorFlowSetNumThreads( number_of_threads ):
    K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads = number_of_threads, inter_op_parallelism_threads = number_of_threads ) ) )


#function to use auc as a keras metric 
def auc( true_labels, predictions, weights = None ):
    auc = tf.metrics.auc( true_labels,  predictions, weights = weights)[1]
    K.get_session().run( tf.local_variables_initializer() )
    return auc


def trainDenseClassificationModel(train_data, train_labels, validation_data, validatation_labels, train_weights = None, validation_weights = None, model_name = 'model', number_of_hidden_layers = 5, units_per_layer = 256, activation = 'relu', optimizer = optimizers.RMSprop(), dropout_first=True, dropout_all=False, dropout_rate = 0.5, num_epochs = 20, batch_size = 128, number_of_threads = 1):

    model = models.Sequential()

    #assume 2D input array
    input_shape = ( len(train_data[-1]), )

    #start with normalization layer to deal with potentially unprocessed data 
    model.add( layers.BatchNormalization( input_shape = input_shape ) )
    
    #add hidden densely connected layers 
    for x in range(number_of_hidden_layers) :
        model.add( layers.Dense( units_per_layer, activation = activation ) )
        if (dropout_first and x == 0) or dropout_all:
            model.add( layers.Dropout( dropout_rate ) )
    
    #output layer
    model.add( layers.Dense(1, activation = 'sigmoid') )

    #print out model configuration
    model.summary()

    #set model objectives
    model.compile(
        optimizer = optimizer,
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )

    #cut off training at convergence and save model with best validation
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor = 'acc',
            patience = 4
        ),
        callbacks.ModelCheckpoint(
            monitor = 'val_acc',
            filepath = model_name + '.h5',
            save_best_only = True
        )
    ]

    #set number of threads to use in training        
    tensorFlowSetNumThreads( number_of_threads ) 

    #train model
    training_history = model.fit(
        train_data,
        train_labels,
        sample_weight = (None if train_weights.size == 0 else train_weights),
        epochs = num_epochs,
        batch_size = batch_size,
        validation_data = (validation_data, validatation_labels, (None if validation_weights.size == 0 else validation_weights) ),
        callbacks = callbacks_list,
        verbose = 2
    )

    #delete weights in saved model to cirumvent a bug in Keras when loading the model
    trained_model_file = h5py.File( model_name + '.h5' , 'r+')
    del trained_model_file['optimizer_weights']
    trained_model_file.close()

    #plot loss and accuracy as a function of epochs
    plotAccuracyComparison( training_history, model_name )
    plotLossComparison( training_history, model_name )

    #to avoid random tensorflow crashes
    K.clear_session()
