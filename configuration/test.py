import os
import sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )
from configuration.Configuration import newConfigurationFromDict
from configuration.LearningAlgorithms import DenseNeuralNetworkConfiguration


if __name__ == '__main__' :
    parameters = {
            'num_hidden_layers' : 512,
            'units_per_layer' : 10,
            'optimizer' : 'Nadam', 
            'learning_rate' : 1,
            'learning_rate_decay' : 1,
            'dropout_first' : False,
            'dropout_all' : False,
            'dropout_rate' : 0.3
            }
    config = newConfigurationFromDict( **parameters ) 
    #config = DenseNeuralNetworkConfiguration( **parameters )
    print( config )

