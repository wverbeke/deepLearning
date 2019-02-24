"""
Define and register specific machine learning algorithms here 
"""

import os
import sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) 
sys.path.insert( 0, main_directory )


#include other parts of framework
from configuration.Configuration import Configuration, registerConfiguration


@registerConfiguration
class DenseNeuralNetworkConfiguration( Configuration ):
    _required_parameters = { 'num_hidden_layers', 'units_per_layer', 'optimizer', 'learning_rate', 'learning_rate_decay', 'dropout_first', 'dropout_all', 'dropout_rate' }

    def _removeRedundancies( self ):
        if self._parameters['dropout_all'] and self._parameters['dropout_first']:
            self._parameters['dropout_first'] = False

        if not( self._parameters['dropout_all'] or self._parameters['dropout_first'] ):
            self._parameters['dropout_rate'] = 0



@registerConfiguration
class GradientBoostedForestConfiguration( Configuration ):
    _required_parameters = { 'number_of_trees', 'learning_rate', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree', 'gamma', 'alpha' }

