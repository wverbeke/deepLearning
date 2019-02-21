"""
Class that collects a numpy array of features, and the corresponding weight for every event 
"""

#import python libraries
import numpy as np
import os

#import ROOT classes 
from ROOT import TFile
from ROOT import TTree

#import keras classes 
#from keras import models
#from keras import optimizers

#import other parts of code 
import sys 
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )

from trainKerasModel import trainDenseClassificationModel
from diagnosticPlotting import *
from configuration.LearningAlgorithms import *
from configuration.Optimizer import Optimizer
#from dataset.Dataset import Dataset 
from dataset.DataCollection import DataCollection


#registry of model training functions 
training_function_registry = {}
def registerTrainingFunction( model_class ):
    def register( func ):
        training_function_registry[model_class] = func 
        return func
    return register 



class ModelTrainingSetup:

    def __init__( self, training_data_configuration ):
            
        #make sure input file exists 
        root_file_name = os.path.join( os.path.dirname(os.path.abspath( __file__) ) , training_data_configuration['root_file_name'] )
        if not os.path.isfile( root_file_name ):
            raise FileNotFoundError('input file does not exist. Give a valid ROOT file!') 

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
        model = models.load_model( configuration.name() + '.h5')
        
        #make predictions 
        signal_training_outputs = model.predict( self.signal_collection.getTrainingSet().getSamples() )
        signal_validation_outputs = model.predict( self.signal_collection.getValidationSet().getSamples() )
        
        background_training_outputs = model.predict( self.background_collection.getTrainingSet().getSamples() )
        background_validation_outputs = model.predict( self.background_collection.getValidationSet().getSamples() )


        #MOVE THIS TO A SEPARATE FUNCTION
        #plot ROC curve and compute ROC integral for validation set 
        eff_signal, eff_background = computeROC(
            signal_validation_outputs, 
            self.signal_collection.getValidationSet().getWeights(), 
            background_validation_outputs,
            self.background_collection.getValidationSet().getWeights(),
            num_points = 1000
        )
        plotROC( eff_signal, eff_background, configuration.name() )
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
