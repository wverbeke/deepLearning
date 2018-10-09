import os
import sys
import argparse

#check if atleast one additional argument is given
if len( sys.argv ) < 2:
    print( 'Error: incorrect number of arguments given to script.')
    print( 'Usage: <python runTraining.py configuration.py>')
    sys.exit()

#read input file 
configuration_file_name = sys.argv[1]
configuration_file = __import__( configuration_file_name.replace('.py', '') )

from Dataset import Data
from jobSubmission import * 
from trainKerasModel import denseModelName

def trainAndEvaluateModel( num_hidden_layers, units_per_layer, learning_rate, dropout_first, dropout_all, dropout_rate):

    #make sure correct path is given for input root file
    root_file_name_full = os.path.join( os.path.dirname(os.path.abspath( __file__) ) , configuration_file.root_file_name )

    #train model
    classification_data = Data(
        root_file_name_full, 
        configuration_file.signal_tree, 
        configuration_file.background_tree, 
        configuration_file.branch_list, 
        configuration_file.weight_branch, 
        configuration_file.validation_fraction, 
        configuration_file.test_fraction, 
        configuration_file.only_positive_weights
    )
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

    #make directory and switch to it in script 
    os.system('mkdir -p output/{}'.format( model_name ) )
    script.write( 'cd output/{}\n'.format( model_name ))

    #run training code 
    training_command = 'python {0} {1}'.format( os.path.realpath(__file__), configuration_file_name )
    training_command += ' {0} {1} {2} {3} {4} {5}'.format( num_hidden_layers, units_per_layer, learning_rate, dropout_first, dropout_all, dropout_rate)

    #pipe output to text files 
    log_file = model_name + '_log.txt'
    error_file = model_name + '_err.txt'
    training_command += ' > {} 2>{} '.format( log_file, error_file) 
    script.write( training_command + '\n')
    script.close()

    #submit script to cluster 
    submitJobScript( 'train_keras_model.sh' )    
    #with open( 'train_keras_model.sh' ) as f :
    #    print( f.read() )
       
 
if __name__ == '__main__' :

    if len( sys.argv ) > 2:    
        parser = argparse.ArgumentParser()
        parser.add_argument('configuration_file_name', type=str)
        parser.add_argument('num_hidden_layers', type=int)
        parser.add_argument('units_per_layer', type=int)
        parser.add_argument('learning_rate', type=float)
        parser.add_argument('dropout_first', type=bool)
        parser.add_argument('dropout_last', type=bool)
        parser.add_argument('dropout_rate', type=float)
        #parser.add_argument('only_positive_weights', type=bool, default=True)
        args = parser.parse_args()
        
        trainAndEvaluateModel( args.num_hidden_layers, args.units_per_layer, args.learning_rate, args.dropout_first, args.dropout_last, args.dropout_rate)        

    else :
        num_networks = len( configuration_file.num_hidden_layers )*len( configuration_file.units_per_layer )*len(  configuration_file.learning_rates )*len(  configuration_file.dropout_first )*len( configuration_file.dropout_all )*len( configuration_file.dropout_rate )

        print( 'Number of neural networks to be trained = {}'.format( num_networks ) )
        if num_networks > 10000:
            print( 'Error : requesting to train more than 10000 neural networks. This will be too much for the T2 cluster to handle.' )
            print( 'Aborting.')
            sys.exit()

        #submit a job for each variation of the neural network parameters
        for num_hidden_layers_var in configuration_file.num_hidden_layers:
            for units_per_layer_var in configuration_file.units_per_layer:
                for learning_rate_var in configuration_file.learning_rates:
                    for dropout_first_var in configuration_file.dropout_first:
                        for dropout_all_var in configuration_file.dropout_all:
                            for dropout_rate_var in configuration_file.dropout_rate:
                                submitTrainingJob(num_hidden_layers_var, units_per_layer_var, learning_rate_var, dropout_first_var, dropout_all_var, dropout_rate_var) 
