from keras import models
from keras import layers
from keras import optimizers 
from keras import callbacks

from treeToArray import *

from ROOT import TFile
import numpy as np

from keras import backend as K


def tensorFlowSetNumThreads( num_threads ):
    K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads = num_threads, inter_op_parallelism_threads = num_threads ) ) )


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
    model_output_name = 'model_{0}hiddenLayers_{1}unitsPerLayer_{2}_learningRate{3}'.format(num_hidden_layers, units_per_layer, activation, learning_rate)  
    model_output_name += ( '_dropoutFirst' if dropoutFirst else '' )
    model_output_name += ( '_dropoutAll' if dropoutAll else '' )
    model_output_name += '.h5'

    #cut off training at convergence and save model with best validation
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor = 'acc',
            patience = 2
        ),
        callbacks.ModelCheckpoint(
            monitor = 'val_acc',
            filepath = model_output_name,
            save_best_only = True
        )
    ]

            
    tensorFlowSetNumThreads( num_threads ) 

    #train model
    model.fit(
        train_data,
        train_labels,
        sample_weight = (None if train_weights.size == 0 else train_weights),
        epochs = num_epochs,
        batch_size = 128,
        validation_data = (validation_data, validatation_labels, (None if validation_weights.size == 0 else validation_weights) ),
        callbacks = callbacks_list
    )


   
if __name__ == '__main__':
    testFile = TFile("ttW_trainingData_new.root")
    testFile.ls()
    signalTree = testFile.Get('signalTree')
    backgroundTree = testFile.Get('bkgTree')
    branchList = [ branch.GetName() for branch in signalTree.GetListOfBranches() ]
    etaBranches = [ name for name in branchList if 'Eta' in name ]
    branchList = [ name for name in branchList if not 'Charge' in name ]
    branchList = [ name for name in branchList if not 'E' in name ]
    branchList += etaBranches
    branchList.remove('_weight')

    training_data_signal = treeToArray( signalTree, branchList )
    training_weights_signal = treeToArray( signalTree, '_weight' )
    training_labels_signal = np.ones( len(training_data_signal) )

    training_data_background = treeToArray( backgroundTree, branchList)
    training_weights_background = treeToArray( backgroundTree, '_weight' )
    training_labels_background = np.zeros( len(training_data_background) )

    training_data = np.concatenate( (training_data_signal, training_data_background) , axis = 0)
    training_weights = np.concatenate( (training_weights_signal, training_weights_background ), axis = 0)
    training_labels = np.concatenate( (training_labels_signal, training_labels_background) , axis = 0)

    indices = range( len( training_data ) )
    np.random.shuffle( indices )

    training_data = training_data[indices] 
    training_weights = training_weights[indices] 
    training_labels = training_labels[indices]
    
    validation_split = 0.5
    split_index = int( len( training_data )*validation_split )
    validation_data = training_data[: split_index ]
    validation_weights = training_weights[: split_index ]
    validation_labels = training_labels[: split_index ]

    training_data = training_data[split_index : ]
    training_weights = training_weights[split_index : ]
    training_labels = training_labels[split_index : ]

    trainDenseClassificationModel( training_data, training_labels, validation_data, validation_labels, training_weights, validation_weights, num_threads = 8, learning_rate = 0.001, dropoutFirst = False)

