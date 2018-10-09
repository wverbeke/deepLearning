#name of input root file, relative to the directory of this script
root_file_name = 'ttW_trainingData_new.root'

#names of trees that contain signal and background events 
signal_tree = 'signalTree'
background_tree = 'bkgTree'

#list of variables to be used in training (corresponding to branches in the tree)
branch_list = [
	'_lPt1', '_lEta1', '_lPhi1',
	'_lPt2', '_lEta2', '_lPhi2',
	'_jetPt1', '_jetEta1', '_jetPhi1', '_jetCSV1',
	'_jetPt2', '_jetEta2', '_jetPhi2', '_jetCSV2',
	'_jetPt3', '_jetEta3', '_jetPhi3', '_jetCSV3',
	'_jetPt4', '_jetEta4', '_jetPhi4', '_jetCSV4',
	'_jetPt5', '_jetEta5', '_jetPhi5', '_jetCSV5',
	'_jetPt6', '_jetEta6', '_jetPhi6', '_jetCSV6',
	'_metPt1', '_metPhi1'
	]

#branch that indicates the event weights 
weight_branch = '_weight'

#use only positive weights in training or not 
only_positive_weights = True

#validation and test fractions
validation_fraction = 0.4
test_fraction = 0.2

#different neural network parameters to test
num_hidden_layers = [2, 3, 4, 5, 6, 7, 8, 9, 10]
units_per_layer = [64, 128, 256, 512]
learning_rates = [0.001, 0.0001, 0.00001]
dropout_first = [False, True]
dropout_all = [False, True] 
dropout_rate = [0.5, 0.3]


import os
import sys
import argparse

from Dataset import Data
from jobSubmission import * 
from trainKerasModel import denseModelName



def trainAndEvaluateModel( num_hidden_layers, units_per_layer, learning_rate, dropout_first, dropout_all, dropout_rate):

    #train model
    classification_data = Data(root_file_name, signal_tree, background_tree, branchList, weight_branch, validation_fraction, test_fraction, only_positive_weights )
    classification_data.trainDenseClassificationModel(
    	num_hidden_layers = num_hidden_layers, 
    	units_per_layer = units_per_layer, 
    	activation = 'relu', 
    	learning_rate = learning_rate, 
    	dropout_first = dropout_first,
    	dropout_all = dropout_all, 
    	dropout_rate = dropout_rate, 
    	num_epochs = 200,
    	num_threads = 1
    )
 

def submitTrainingJob(num_hidden_layers, units_per_layer, learning_rate, dropout_first, dropout_all, dropout_rate):

    #make script that will be submitted 
    script = initializeJobScript('train_keras_model.sh')

    #make name of model that will be trained 
    model_name = denseModelName(num_hidden_layers, units_per_layer, 'relu', learning_rate, dropout_first, dropout_all, dropout_rate)

    #make directory and switch to it 
    os.system('mkdir -p output/{}'.format( model_name ) )
    
    #switch to this directory in the script 
    script.write( 'cd output/{}'.format( model_name ) )


    training_command = 'python {0}'.format( os.path.realpath(__file__) )
    training_command += ' {0} {1} {2} {3} {4} {5}'.format( num_hidden_layers, units_per_layer, learning_rate, dropout_first, dropout_all, dropout_rate)
    script.write( training_command + '\n')
    script.close()

    #submit script to cluster 
    #submitJobScript( 'train_keras_model.sh' )    
    with open( 'train_keras_model.sh' ) as f :
        print( f.read() )
       
 
if __name__ == '__main__' :

    if len( sys.argv ) > 1:    
        parser = argparse.ArgumentParser()
        parser.add_argument('num_hidden_layers', type=int)
        parser.add_argument('units_per_layer', type=int)
        parser.add_argument('learning_rate', type=float)
        parser.add_argument('dropout_first', type=bool)
        parser.add_argument('dropout_rate', type=float)
        parset.add_argument('only_positive_weights', type=bool, default=True)
        args = parser.parse_args()
        
        trainAndEvaluateModel( args.num_hidden_layers, args.units_per_layer, args.learning_rate, args.dropoutFirst, args.dropoutRate)        

    else:
        num_networks = len( num_hidden_layers )*len( units_per_layer )*len( learning_rates )*len( dropout_first )*len( dropout_all )*len( dropout_rate )
        print( 'Number of neural networks to be trained = {}'.format( num_networks ) )
        if num_networks > 10000:
            print( 'Error : requesting to train more than 10000 neural networks. This will be too much for the T2 cluster to handle.' )
            print( 'Aborting.')
            sys.exit()

        #submit a job for each variation of the neural network parameters
        for num_hidden_layers_var in num_hidden_layers:
            for units_per_layer_var in units_per_layer:
                for learning_rate_var in learning_rates:
                    for dropout_first_var in dropout_first:
                        for dropout_all_var in dropout_all:
                            for dropout_rate_var in dropout_rate:
                                submitTrainingJob(num_hidden_layers_var, units_per_layer_var, learning_rate_var, dropout_first_var, dropout_all_var, dropout_rate_var) 
                        
