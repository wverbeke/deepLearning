"""
Class that collects a numpy array of features, and the corresponding weight for every event 
"""

#import other parts of code 
from treeToArray import treeToArray
from trainKerasModel import trainDenseClassificationModel
from diagnosticPlotting import *

#import python libraries
import numpy as np
import os.path

#import ROOT classes 
from ROOT import TFile
from ROOT import TTree

#import keras classes 
from keras import models


def randomlyShuffledIndices( array ):
    indices = list( range( len( array ) ) )
    np.random.shuffle( indices )
    return indices 


class Dataset:
    def __init__(self, samples, weights, labels):

        #make sure each samples has a weight and vice-versa
        if len(samples) != len(weights) or len(samples) != len(labels):
            print('Error in Dataset::__init__ : sample, weight and label arrays must have equal length!')
            return 

        self.samples = samples 
        self.weights = weights  
        self.labels = labels

    def __len__(self):
        return len( self.samples )

    def getSamples(self):
        return self.samples
    
    def getWeights(self):
        return self.weights

    def getLabels(self):
        return self.labels

    def __add__ (self, rhs):
        samples = np.concatenate( (self.samples, rhs.samples), axis = 0)
        weights = np.concatenate( (self.weights, rhs.weights), axis = 0)
        labels = np.concatenate( (self.labels, rhs.labels), axis = 0)
        return Dataset(samples, weights, labels)


def concatenateAndShuffleSets( lhs_dataset, rhs_dataset):
    merged_set = lhs_dataset + rhs_dataset 
    indices = randomlyShuffledIndices( merged_set )
    merged_set.samples = merged_set.getSamples()[indices]
    merged_set.weights = merged_set.getWeights()[indices]
    merged_set.labels = merged_set.getLabels()[indices]
    return merged_set 


class DataCollection:
    def __init__(self, data_training, data_validation, data_testing):
        self.data_training = data_training
        self.data_validation = data_validation
        self.data_testing = data_testing 


    def __init__(self, tree, branch_names, weight_name, validation_fraction, test_fraction, is_signal, only_positive_weights):

        #test if sensible input is given
        if (validation_fraction + test_fraction ) >= 1:
            print('Error in DataCollection::__init__ : validation and test fractions sum to a value greater or equal to 1!')
            return

        #read total dataset from tree, and only retain positive weight events if asked 
        reading_cut = '{}>0'.format(weight_name) if only_positive_weights else ''
        samples_total = treeToArray( tree, branch_names, reading_cut)
        weights_total = treeToArray( tree, weight_name, reading_cut )
        num_samples = len(samples_total)
        labels_total = np.ones( num_samples ) if is_signal else np.zeros( num_samples ) 

        #randomly shuffle the datasets to prevent any structure
        indices = randomlyShuffledIndices( samples_total )
        samples_total = samples_total[indices]
        weights_total = weights_total[indices]

        #split training/validation and test sets
        max_index_training = int( num_samples*( 1 - validation_fraction - test_fraction ) )
        max_index_validation = int( num_samples*( 1 - test_fraction ) )

        self.data_training = Dataset( samples_total[:max_index_training], weights_total[:max_index_training], labels_total[:max_index_training]) 
        self.data_validation = Dataset( samples_total[max_index_training:max_index_validation], weights_total[max_index_training:max_index_validation], labels_total[max_index_training:max_index_validation])
        self.data_testing = Dataset( samples_total[max_index_training:], weights_total[max_index_training:], labels_total[max_index_training:])


    def getTrainingSet(self):
        return self.data_training

    
    def getValidationSet(self):
        return self.data_validation

    
    def getTestSet(self):
        return self.data_test 



class Data:
    def __init__(self, signal_collection, background_collection):
        self.signal_collection = signal_collection
        self.background_collection = background_collection


    def __init__(self, file_name, tree_signal_name, tree_background_name, branch_names, weight_name, validation_fraction, test_fraction, only_positive_weights = True):
            
        #make sure input file exists 
        if not os.path.isfile( file_name ):
            print('Error in Data::__init__ input file does not exist. Give a valid ROOT file!')
            return

        #get trees from file
        root_file = TFile(file_name)
        tree_signal = root_file.Get(tree_signal_name)
        tree_background = root_file.Get(tree_background_name)

        #use trees to initialize data
        self.signal_collection = DataCollection( tree_signal, branch_names, weight_name, validation_fraction, test_fraction, True, only_positive_weights)
        self.background_collection = DataCollection( tree_background, branch_names, weight_name, validation_fraction, test_fraction, False, only_positive_weights)


    def trainDenseClassificationModel(self, num_hidden_layers = 5, units_per_layer = 256, activation = 'relu', learning_rate = 0.0001, dropout_first=True, dropout_all=False, dropout_rate = 0.5, num_epochs = 20, num_threads = 1):
        
        #make shuffled training and validation sets 
        training_data = concatenateAndShuffleSets( self.signal_collection.getTrainingSet(), self.background_collection.getTrainingSet() )
        validation_data = concatenateAndShuffleSets( self.signal_collection.getTrainingSet(), self.background_collection.getTrainingSet() )
        
        #train classifier 
        model_name = trainDenseClassificationModel(
            training_data.getSamples(), training_data.getLabels(), validation_data.getSamples(), validation_data.getLabels(), 
            train_weights = training_data.getWeights(), validation_weights = validation_data.getWeights(), 
            num_hidden_layers = num_hidden_layers, 
            units_per_layer = units_per_layer, 
            activation = activation, 
            learning_rate = learning_rate, 
            dropout_first = dropout_first, 
            dropout_all = dropout_all, 
            dropout_rate = dropout_rate, 
            num_epochs = num_epochs, 
            num_threads = num_threads
        )
        
        ##load trained classifier 
        model = models.load_model(model_name + '.h5')
        
        #make predictions 
        signal_training_outputs = model.predict( self.signal_collection.getTrainingSet().getSamples() )
        signal_validation_outputs = model.predict( self.signal_collection.getValidationSet().getSamples() )
        
        background_training_outputs = model.predict( self.background_collection.getTrainingSet().getSamples() )
        background_validation_outputs = model.predict( self.background_collection.getValidationSet().getSamples() )
        
        #plot ROC curve and compute ROC integral for validation set 
        eff_signal, eff_background = computeROC(
            signal_validation_outputs, 
            self.signal_collection.getValidationSet().getWeights(), 
            background_validation_outputs,
            self.background_collection.getValidationSet().getWeights(),
            num_points = 1000
        )
        plotROC( eff_signal, eff_background, model_name)
        auc = areaUndeCurve(eff_signal, eff_background )
        print('#####################################################')
        print('validation set ROC integral (AUC) = {:.3f}'.format(auc) )
        print('#####################################################')
        
        #compare output shapes 
        plotOutputShapeComparison( signal_training_outputs, self.signal_collection.getTrainingSet().getWeights(),
        	background_training_outputs, self.background_collection.getTrainingSet().getWeights(),	
        	signal_validation_outputs, self.signal_collection.getValidationSet().getWeights(),
        	background_validation_outputs, self.background_collection.getValidationSet().getWeights(),
        	model_name
        )
        
         
        

