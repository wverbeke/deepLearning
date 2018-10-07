"""
Class that collects a numpy array of features, and the corresponding weight for every event 
"""

#import other parts of code 
from treeToArray import treeToArray

#import python libraries
import numpy as np


class Dataset:
    def __init__(self, samples, weights):

        #make sure each samples has a weight and vice-versa
        if len(samples) != len(weights):
            print('Error in Dataset::__init__ : sample and weight arrays must have equal length!')
            return 

        self.samples = samples 
        self.weights = weights  
    
    def getDataset():
        return self.samples, self.weights



class DataCollection:
    def __init__(self, data_training, data_validation, data_testing):
        self.data_training = data_training
        self.data_validation = data_validation
        self.data_testing = data_testing 


    def __init__(self, tree, branch_names, weight_name, validation_fraction, test_fraction):

        #test if sensible input is given
        if (validation_fraction + test_fraction ) >= 1:
            print('Error in DataCollection::__init__ : validation and test fractions sum to a value greater or equal to 1!')
            return

        #read total dataset from tree
        samples_total = treeToArray( tree, branch_names )
        weights_total = treeToArray( tree, weight_name )

        #randomly shuffle the datasets to prevent any structure
        num_samples = len(samples_total)
        indices = list( range( num_samples ) ) #in python 3 list does not return a list 
        np.random.shuffle( indices )
        samples_total = samples_total[indices]
        weights_total = weights_total[indices]

        #split training/validation and test sets
        max_index_training = int( num_samples*( 1 - validation_fraction - test_fraction ) )
        max_index_validation = int( num_samples*( 1 - test_fraction ) )

        self.data_training = Dataset( samples_total[:max_index_training], weights_total[:max_index_training] ) 
        self.data_validation = Dataset( samples_total[max_index_training:max_index_validation], weights_total[max_index_training:max_index_validation] )
        self.data_testing = Dataset( samples_total[max_index_training:], weights_total[max_index_training:] )


    def getTrainingSet():
        return self.data_training

    
    def getValidationSet():
        return self.data_validation

    
    def getTestSet():
        return self.data_test 
