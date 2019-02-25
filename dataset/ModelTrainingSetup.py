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
from keras import models

#import xgboost 
import xgboost as xgb

#import other parts of code 
import sys 
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )

from modelTraining.trainKerasModel import trainDenseClassificationModel, auc 
from modelTraining.trainXGBoostModel import trainGradientBoostedForestClassificationModel
from diagnosticPlotting import *
from configuration.LearningAlgorithms import *
from configuration.Optimizer import Optimizer
from dataset.Dataset import concatenateAndShuffleDatasets
from dataset.DataCollection import DataCollection


#registry of model training functions 
training_function_registry = {}
def registerTrainingFunction( model_class ):
    def register( func ):
        training_function_registry[model_class] = func 
        return func
    return register 



#evaluate a model from its outputs and weights for signal and background 
def rocAndAUC( signal_dataset, background_dataset, model_name ):
	
	#plot ROC curve and compute ROC integral for validation set 
	eff_signal, eff_background = computeROC(
	    signal_dataset.outputs,
	    signal_dataset.weights,
	    background_dataset.outputs,
	    background_dataset.weights,
	    num_points = 10000
	)
	plotROC( eff_signal, eff_background, model_name )
	auc = areaUnderCurve(eff_signal, eff_background )
	print('#####################################################')
	print('validation set ROC integral (AUC) = {:.5f}'.format(auc) )
	print('#####################################################')
	

def compareOutputShapes( signal_training_dataset, signal_validation_dataset, background_training_dataset, background_validation_dataset, model_name):

	#compare output shapes 
	plotOutputShapeComparison( 
        signal_training_dataset.outputs, signal_training_dataset.weights,
	    background_training_dataset.outputs, background_training_dataset.weights,
        signal_validation_dataset.outputs, signal_validation_dataset.weights,
	    background_validation_dataset.outputs, background_validation_dataset.weights,
	    model_name
	)



class ModelTrainingSetup:

    def __init__( self, training_data_configuration ):
            
        #to ensure reproducible splitting of the datasets
        np.random.seed(1)

        #make sure input file exists 
        root_file_name = os.path.join( main_directory , training_data_configuration['root_file_name'] )
        if not os.path.isfile( root_file_name ):
            raise OSError('input file does not exist. Give a valid ROOT file!') 

        #get trees from file
        root_file = TFile( root_file_name )
        tree_signal = root_file.Get( training_data_configuration['signal_tree_name'] )
        tree_background = root_file.Get( training_data_configuration['background_tree_name'] )

        #use trees to initialize data
        self.__signal_collection = DataCollection( 
            tree_signal, 
            training_data_configuration['list_of_branches'], 
            training_data_configuration['validation_fraction'],
            training_data_configuration['test_fraction'], 
            True, 
            training_data_configuration['weight_branch'], 
            training_data_configuration['only_positive_weights']
        )
        self.__background_collection = DataCollection( 
            tree_background,
            training_data_configuration['list_of_branches'], 
            training_data_configuration['validation_fraction'],
            training_data_configuration['test_fraction'], 
            False, 
            training_data_configuration['weight_branch'], 
            training_data_configuration['only_positive_weights']
        )
    
        self.__number_of_threads = training_data_configuration['number_of_threads']
        self.__feature_names = training_data_configuration['list_of_branches']


    #make shuffled training and validation sets 
    def trainingAndValidationSets( self ):
        training_data = concatenateAndShuffleDatasets( self.__signal_collection.training_set, self.__background_collection.training_set )
        validation_data = concatenateAndShuffleDatasets( self.__signal_collection.validation_set, self.__background_collection.validation_set )
        return training_data, validation_data

    
    #plot ROC curve, compute AUC and plot shape comparison after adding model predictions to the datasets 
    def plotROCAndShapeComparison( self, model_name ):
        rocAndAUC( self.__signal_collection.validation_set, self.__background_collection.validation_set, model_name )
        compareOutputShapes(
            self.__signal_collection.training_set,
            self.__signal_collection.validation_set,
            self.__background_collection.training_set,
            self.__background_collection.validation_set,
            model_name
        )


    def trainAndEvaluateModel( self, configuration ):
        for classKey in training_function_registry:
            if isinstance( configuration, classKey ):
                training_function_registry[classKey](self, configuration)
        self.plotROCAndShapeComparison( configuration.name() )



    """add functions to train and evaluate a specific machine learning algorithm below, and register them with the correct configuration class"""

    #train dense neural network in Keras with TensorFlow backend
    @registerTrainingFunction( DenseNeuralNetworkConfiguration )
    def trainDenseNeuralNetworkClassificationModel( self, configuration ):
        
        training_data, validation_data = self.trainingAndValidationSets()

        #make keras optimizer object 
        keras_optimizer = Optimizer( configuration['optimizer'], configuration['learning_rate'], configuration['learning_rate_decay'] ).kerasOptimizer()

        #train classifier 
        trainDenseClassificationModel(
            training_data.samples, training_data.labels, validation_data.samples, validation_data.labels, 
            train_weights = training_data.weights, validation_weights = validation_data.weights, 
            model_name = configuration.name(), 
            num_hidden_layers = configuration['num_hidden_layers'],
            units_per_layer = configuration['units_per_layer'],
            activation = 'relu', 
            optimizer = keras_optimizer,
            dropout_first = configuration['dropout_first'],
            dropout_all = configuration['dropout_all'],
            dropout_rate = configuration['dropout_rate'],
            num_epochs = 1, 
            number_of_threads = self.__number_of_threads
        )

        #load trained classifier 
        model = models.load_model( configuration.name() + '.h5', custom_objects = {'auc':auc})

        #make predictions 
        self.__signal_collection.training_set.addOutputs( model.predict( self.__signal_collection.training_set.samples ) )
        self.__signal_collection.validation_set.addOutputs( model.predict( self.__signal_collection.validation_set.samples ) )

        self.__background_collection.training_set.addOutputs( model.predict( self.__background_collection.training_set.samples ) )
        self.__background_collection.validation_set.addOutputs( model.predict( self.__background_collection.validation_set.samples ) )

    
    #train gradient boosted forest in XGBoost
    @registerTrainingFunction( GradientBoostedForestConfiguration )
    def trainGradientBoostedForestClassificationModel( self, configuration ):

        training_data, validation_data = self.trainingAndValidationSets()
    
        trainGradientBoostedForestClassificationModel(
            training_data.samples, training_data.labels, train_weights = training_data.weights, 
            feature_names = self.__feature_names,
            model_name = configuration.name(),
            number_of_trees = configuration['number_of_trees'],
            learning_rate = configuration['learning_rate'],
            max_depth = configuration['max_depth'],
            min_child_weight = configuration['min_child_weight'],
            subsample = configuration['subsample'],
            colsample_bytree = configuration['colsample_bytree'],
            gamma = configuration['gamma'],
            alpha = configuration['alpha'],
            number_of_threads = self.__number_of_threads
        )

        #load trained classifier 
        model = xgb.Booster()
        model.load_model( configuration.name() + '.bin' )

        #make xgboost DMatrices for predictions 
        signal_training_matrix = xgb.DMatrix( self.__signal_collection.training_set.samples, label = self.__signal_collection.training_set.labels, nthread = self.__number_of_threads)
        signal_validation_matrix = xgb.DMatrix( self.__signal_collection.validation_set.samples, label = self.__signal_collection.validation_set.labels, nthread = self.__number_of_threads)

        background_training_matrix = xgb.DMatrix( self.__background_collection.training_set.samples, label = self.__background_collection.training_set.labels, nthread = self.__number_of_threads)
        background_validation_matrix = xgb.DMatrix( self.__background_collection.validation_set.samples, label = self.__background_collection.validation_set.labels, nthread = self.__number_of_threads)
        
        #make predictions 
        self.__signal_collection.training_set.addOutputs( model.predict( signal_training_matrix ) )
        self.__signal_collection.validation_set.addOutputs( model.predict( signal_validation_matrix ) )

        self.__background_collection.training_set.addOutputs( model.predict( background_training_matrix ) )
        self.__background_collection.validation_set.addOutputs( model.predict( background_validation_matrix ) )
