"""
Class that collects a numpy array of features, and the corresponding weight for every event 
"""

#import other parts of code 
from treeToArray import treeToArray

#import python libraries
import numpy as np
import os.path

#import ROOT classes 
from ROOT import TFile
from ROOT import TTree


class Dataset:
    def __init__(self, samples, weights, labels):

        #make sure each samples has a weight and vice-versa
        if len(samples) != len(weights):
            print('Error in Dataset::__init__ : sample and weight arrays must have equal length!')
            return 

        self.samples = samples 
        self.weights = weights  
        self.labels = labels
    
    def getDataset():
        return self.samples, self.weights, self.labels



class DataCollection:
    def __init__(self, data_training, data_validation, data_testing):
        self.data_training = data_training
        self.data_validation = data_validation
        self.data_testing = data_testing 


    def __init__(self, tree, branch_names, weight_name, validation_fraction, test_fraction, is_signal):

        #test if sensible input is given
        if (validation_fraction + test_fraction ) >= 1:
            print('Error in DataCollection::__init__ : validation and test fractions sum to a value greater or equal to 1!')
            return

        #read total dataset from tree
        samples_total = treeToArray( tree, branch_names )
        weights_total = treeToArray( tree, weight_name )
        num_samples = len(samples_total)
        labels_total = np.ones( num_samples ) if is_signal else np.zeros( num_samples ) 

        #randomly shuffle the datasets to prevent any structure
        indices = list( range( num_samples ) ) #in python 3 list does not return a list 
        np.random.shuffle( indices )
        samples_total = samples_total[indices]
        weights_total = weights_total[indices]

        #split training/validation and test sets
        max_index_training = int( num_samples*( 1 - validation_fraction - test_fraction ) )
        max_index_validation = int( num_samples*( 1 - test_fraction ) )

        self.data_training = Dataset( samples_total[:max_index_training], weights_total[:max_index_training], labels[:max_index_training]) 
        self.data_validation = Dataset( samples_total[max_index_training:max_index_validation], weights_total[max_index_training:max_index_validation], labels[max_index_training:max_index_validation])
        self.data_testing = Dataset( samples_total[max_index_training:], weights_total[max_index_training:], labels[max_index_training:])


    def getTrainingSet():
        return self.data_training

    
    def getValidationSet():
        return self.data_validation

    
    def getTestSet():
        return self.data_test 



class Data:
    def __init__(self, signal_collection, background_collection):
        self.signal_collection = signal_collection
        self.background_collection = background_collection

    def __init__(self, tree_signal, tree_background, branch_names, weight_name, validation_fraction, test_fraction):
        self.signal_collection = DataCollection( tree_signal, branch_names, weight_name, validation_fraction, test_fraction, True)
        self.background_collection = DataCollection( tree_background, branch_names, weight_name, validation_fraction, test_fraction, False)

    def __init__(self, file_name, tree_signal_name, tree_background_name, branch_names, weight_name, validation_fraction, test_fraction):
            
        #make sure input file exists 
        if not os.path.isfile( fileName ):
            print('Error in Data::__init__ input file does not exist. Give a valid ROOT file!')
            return

        #get trees from file
        root_file = TFile(file_name)
        tree_signal = root_file.Get(tree_signal_name)
        tree_background = root_file.Get(tree_background_name)

        #use trees to initialize data
        self.__init__(tree_signal, tree_background, branch_names, weight_name, validation_fraction, test_fraction)
        
        

