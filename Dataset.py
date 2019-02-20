"""
Class that collects a numpy array of features, and the corresponding weight for every event 
"""

#import python libraries
import numpy as np
import os.path

#import ROOT classes 
from ROOT import TFile
from ROOT import TTree

#import keras classes 
from keras import models
from keras import optimizers

#import other parts of code 
from treeToArray import treeToArray
from trainKerasModel import trainDenseClassificationModel
from diagnosticPlotting import *
from configuration.LearningAlgorithms import *
from configuration.Optimizer import Optimizer


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



#registry of model training functions 
training_function_registry = {}
def registerTrainingFunction( model_class ):
    def register( func ):
        training_function_registry[model_class] = func 
        return func
    return register 


class Data:
    def __init__(self, signal_collection, background_collection):
        self.signal_collection = signal_collection
        self.background_collection = background_collection


    #def __init__(self, file_name, tree_signal_name, tree_background_name, branch_names, weight_name, validation_fraction, test_fraction, only_positive_weights = True):
    def __init__( self, training_data_configuration ):
            
        #make sure input file exists 
        root_file_name = os.path.join( os.path.dirname(os.path.abspath( __file__) ) , training_data_configuration['root_file_name'] )
        if not os.path.isfile( root_file_name ):
            print('Error in Data::__init__ input file does not exist. Give a valid ROOT file!')
            return

        #get trees from file
        root_file = TFile( root_file_name )
        tree_signal = root_file.Get( training_data_configuration['signal_tree_name'] )
        tree_background = root_file.Get( training_data_configuration['background_tree_name'] )

        #use trees to initialize data
        self.signal_collection = DataCollection( 
            tree_signal, 
            training_data_configuration['list_of_branches'], 
            training_data_configuration['weight_branch'], 
            training_data_configuration['validation_fraction'],
            training_data_configuration['test_fraction'], 
            True, 
            training_data_configuration['only_positive_weights']
        )
        self.background_collection = DataCollection( 
            tree_background,
            training_data_configuration['list_of_branches'], 
            training_data_configuration['weight_branch'], 
            training_data_configuration['validation_fraction'],
            training_data_configuration['test_fraction'], 
            False, 
            training_data_configuration['only_positive_weights']
        )


    def trainAndEvaluateModel( self, configuration ):
        for classKey in training_function_registry:
            if isinstance( configuration, classKey ):
                return training_function_registry[classKey](self, configuration)


    @registerTrainingFunction( DenseNeuralNetworkConfiguration )
    #def trainDenseClassificationModel(self, model_name = 'model', num_hidden_layers = 5, units_per_layer = 256, activation = 'relu', optimizer = optimizers.RMSprop(), dropout_first=True, dropout_all=False, dropout_rate = 0.5, num_epochs = 20, num_threads = 1):
    def trainDenseClassificationModel(self, configuration):
        
        #make shuffled training and validation sets 
        training_data = concatenateAndShuffleSets( self.signal_collection.getTrainingSet(), self.background_collection.getTrainingSet() )
        validation_data = concatenateAndShuffleSets( self.signal_collection.getTrainingSet(), self.background_collection.getTrainingSet() )

        #make optimizer object 
        keras_optimizer = Optimizer( configuration['optimizer'], configuration['learning_rate'], configuration['learning_rate_decay'] ).kerasOptimizer()


        #train classifier 
        trainDenseClassificationModel(
            training_data.getSamples(), training_data.getLabels(), validation_data.getSamples(), validation_data.getLabels(), 
            train_weights = training_data.getWeights(), validation_weights = validation_data.getWeights(), 
            model_name = configuration.name(), 
            num_hidden_layers = configuration['num_hidden_layers'],
            units_per_layer = configuration['units_per_layer'],
            activation = 'relu', 
            optimizer = keras_optimizer,
            dropout_first = configuration['dropout_first'],
            dropout_all = configuration['dropout_all'],
            dropout_rate = configuration['dropout_rate'],
            num_epochs = 500, 
            num_threads = 4
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
        auc = areaUnderCurve(eff_signal, eff_background )
        print('#####################################################')
        print('validation set ROC integral (AUC) = {:.5f}'.format(auc) )
        print('#####################################################')
        
        #compare output shapes 
        plotOutputShapeComparison( signal_training_outputs, self.signal_collection.getTrainingSet().getWeights(),
        	background_training_outputs, self.background_collection.getTrainingSet().getWeights(),	
        	signal_validation_outputs, self.signal_collection.getValidationSet().getWeights(),
        	background_validation_outputs, self.background_collection.getValidationSet().getWeights(),
        	model_name
        )
