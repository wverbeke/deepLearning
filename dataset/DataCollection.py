"""
Class that collects a numpy array of features, and the corresponding weight for every event 
"""

#import python libraries
import numpy as np

#import other parts of framework
import os
import sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )
from dataset.Dataset import Dataset 
from treeToArray import treeToArray



class DataCollection:

    def __init__( self, tree, branch_names, validation_fraction, test_fraction, is_signal, weight_name = None, only_positive_weights = True, parameter_names = None ):

        #test if sensible input is given
        if ( validation_fraction + test_fraction ) >= 1:
            raise ValueError( 'validation and test fractions sum to a value greater or equal to 1!' )

        #read total dataset from tree, and only retain positive weight events if asked 
        reading_cut = '{}>0'.format( weight_name ) if ( only_positive_weights and not weight_name is None ) else ''
        samples_total = treeToArray( tree, branch_names, reading_cut)
        number_of_samples = len( samples_total )
        weights_total = treeToArray( tree, weight_name, reading_cut ) if ( not weight_name is None ) else np.ones( number_of_samples )
    
        #add parameters if requested
        parameters_total = None
        if parameter_names is not None:
            parameters_total = treeToArray( tree, parameter_names, reading_cut )

        #labels to be predicted by the model. 1 for signal and 0 for background
        labels_total = np.ones( number_of_samples ) if is_signal else np.zeros( number_of_samples ) 

        #build the dataset
        total_dataset = Dataset( samples_total, weights_total, labels_total, None, parameters_total )

        #randomly shuffle the dataset to prevent structure before splitting 
        total_dataset.shuffle()

        #split training/validation and test sets
        max_index_training = int( number_of_samples*( 1 - validation_fraction - test_fraction ) )
        max_index_validation = int( number_of_samples*( 1 - test_fraction ) )
        
        self.__training_set = total_dataset[:max_index_training]
        self.__validation_set = total_dataset[max_index_training:max_index_validation]
        self.__test_set = total_dataset[max_index_validation:]

        #avoid large numerical scales in the weights during training
        if weight_name is not None :

            #scaling is strictly only necessary for training set, HOWEVER it is better to also scale the validation set to have losses of the same scale
            weight_scale_factor = 1. / np.mean( self.__training_set.weights )
            self.__training_set.scaleWeights( weight_scale_factor )
            self.__validation_set.scaleWeights( weight_scale_factor )


    @property
    def training_set( self ):
        return self.__training_set

    
    @property
    def validation_set( self ):
        return self.__validation_set

    
    @property
    def test_set( self ):
        return self.__test_set


    #verify that the DataCollection has parametrization
    def isParametric( self ):
        return self.__training_set.isParametric()
