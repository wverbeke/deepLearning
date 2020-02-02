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

from modelTraining.trainKerasModel import KerasClassifierTrainer
from modelTraining.trainXGBoostModel import trainGradientBoostedForestClassificationModel
from diagnosticPlotting import *
from configuration.LearningAlgorithms import *
from configuration.Optimizer import Optimizer
from configuration.Activation import Activation
from dataset.Dataset import concatenateAndShuffleDatasets
from dataset.DataCollection import DataCollection
from dataset.CombinedDataCollection import CombinedDataCollection

#temporary, remove
from configuration.InputReader import *


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
        np.random.seed(42)

        #make sure input file exists 
        root_file_name = os.path.join( main_directory , training_data_configuration['root_file_name'] )
        if not os.path.isfile( root_file_name ):
            raise OSError('input file does not exist. Give a valid ROOT file!') 

        #get trees from file
        root_file = TFile( root_file_name )
        tree_signal = root_file.Get( training_data_configuration['signal_tree_name'] )
        tree_background = root_file.Get( training_data_configuration['background_tree_name'] )

        #use trees to initialize data
        signal_collection = DataCollection( 
            tree_signal, 
            training_data_configuration['list_of_branches'], 
            training_data_configuration['validation_fraction'],
            training_data_configuration['test_fraction'], 
            True, 
            training_data_configuration['weight_branch'], 
            training_data_configuration['only_positive_weights'],
            training_data_configuration['signal_parameters']
        )
        background_collection = DataCollection( 
            tree_background,
            training_data_configuration['list_of_branches'], 
            training_data_configuration['validation_fraction'],
            training_data_configuration['test_fraction'], 
            False, 
            training_data_configuration['weight_branch'], 
            training_data_configuration['only_positive_weights']
        )

        #combined collection that stores signal and background datasets
        self.__combined_collection = CombinedDataCollection( signal_collection, background_collection )

        #store the configuration object 
        self.__training_data_configuration = training_data_configuration

        ##number of threads to use
        #self.__number_of_threads = training_data_configuration['number_of_threads']

        ##store number of features for feature importance plots in XGBoost
        #self.__feature_names = training_data_configuration['list_of_branches']


    #plot ROC curve, compute AUC and plot shape comparison after adding model predictions to the datasets 
    def plotROCAndShapeComparison( self, model_name ):
        rocAndAUC( self.__combined_collection.signal_validation_set, self.__combined_collection.background_validation_set, model_name )
        compareOutputShapes(
            self.__combined_collection.signal_training_set,
            self.__combined_collection.signal_validation_set,
            self.__combined_collection.background_training_set,
            self.__combined_collection.background_validation_set,
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
        
        #training_data, validation_data = self.trainingAndValidationSets()

        #make keras optimizer object 
        keras_optimizer = Optimizer( configuration['optimizer'], configuration['learning_rate'], configuration['learning_rate_decay'] ).kerasOptimizer()

        #retrieve keras activation layer class
        keras_activation_layer = Activation( configuration['activation'] ).kerasActivationLayer()

		#input shape
        number_of_inputs = len( self.__training_data_configuration['list_of_branches'] )
        if ( self.__training_data_configuration['signal_parameters'] is not None ) and ( not self.__training_data_configuration[ 'parameter_shortcut_connection' ] ):
            number_of_inputs += len( self.__training_data_configuration['signal_parameters'] )
        input_shape = ( number_of_inputs, )

        #build the model 
        model_builder = KerasClassifierTrainer( configuration.name() )
        if self.__training_data_configuration[ 'parameter_shortcut_connection' ]:
            model_builder.buildDenseClassifierParameterShortCut(
                sample_input_shape = input_shape,
                parameter_input_shape = ( len( self.__training_data_configuration['signal_parameters'] ), ),
            	number_of_hidden_layers = configuration['number_of_hidden_layers'], 
                units_per_layer = configuration['units_per_layer'], 
                activation_layer = keras_activation_layer, 
                dropout_first = configuration['dropout_first'],
                dropout_all = configuration['dropout_all'],
                dropout_rate = configuration['dropout_rate'],
                batchnorm_first = configuration['batchnorm_first'],
                batchnorm_hidden = configuration['batchnorm_hidden'],
                batchnorm_before_activation = configuration['batchnorm_before_activation'],
            )

        else:
            model_builder.buildDenseClassifier(
				input_shape = input_shape,
                number_of_hidden_layers = configuration['number_of_hidden_layers'], 
                units_per_layer = configuration['units_per_layer'], 
                activation_layer = keras_activation_layer, 
				dropout_first = configuration['dropout_first'],
				dropout_all = configuration['dropout_all'],
				dropout_rate = configuration['dropout_rate'],
				batchnorm_first = configuration['batchnorm_first'],
				batchnorm_hidden = configuration['batchnorm_hidden'],
				batchnorm_before_activation = configuration['batchnorm_before_activation'],
			)

        #train the model 
        if self.__training_data_configuration['signal_parameters'] is None:
            model_builder.trainModelArrays(
            	train_data = self.__combined_collection.training_set().samples,
            	train_labels = self.__combined_collection.training_set().labels,
            	validation_data = self.__combined_collection.validation_set().samples,
                validation_labels = self.__combined_collection.validation_set().labels,
            	train_weights = self.__combined_collection.training_set().weights,
                validation_weights = self.__combined_collection.validation_set().weights,
                optimizer = keras_optimizer,
                number_of_epochs = configuration['number_of_epochs'],
                batch_size = configuration['batch_size'],
                number_of_threads = self.__training_data_configuration['number_of_threads']
            )

        else:

            #make new generator in case of shortcut connection
            if self.__training_data_configuration['parameter_shortcut_connection']:
                number_of_parameters = len( self.__training_data_configuration['signal_parameters'] )
                def shortcutGenerator( parametric_sample_generator ):
                	gen = parametric_sample_generator
                	for entry in gen:
                		samples, labels, weights = entry
                
                		#split samples and parameters
                		pars = samples[:, -number_of_parameters:]
                		samples = samples[:, :-number_of_parameters]
                
                		yield [samples, pars], labels, weights
                
                training_generator = shortcutGenerator( self.__combined_collection.trainingGenerator( configuration['batch_size'] ) )
                validation_generator = shortcutGenerator( self.__combined_collection.validationGenerator( configuration['batch_size'] ) )
            
            
            else:
                training_generator = self.__combined_collection.trainingGenerator( configuration['batch_size'] )
                validation_generator = self.__combined_collection.validationGenerator( configuration['batch_size'] )

            model_builder.trainModelGenerators(
                train_generator = training_generator,
                train_steps_per_epoch = self.__combined_collection.numberOfTrainingBatches( configuration['batch_size'] ),
                validation_generator = validation_generator,
                validation_steps_per_epoch = self.__combined_collection.numberOfValidationBatches( configuration['batch_size'] ),
                optimizer = keras_optimizer,
                number_of_epochs = configuration['number_of_epochs'],
                number_of_threads = self.__training_data_configuration['number_of_threads']
            )


        #load trained classifier 
        #model = models.load_model( configuration.name() + '.h5', custom_objects = {'auc':auc})
        model = models.load_model( configuration.name() + '.h5' )

        #make predictions 
        if self.__training_data_configuration['parameter_shortcut_connection']:
            self.__combined_collection.signal_training_set.addOutputs( model.predict( [ self.__combined_collection.signal_training_set.samples, self.__combined_collection.signal_training_set.parameters ] ) )
            self.__combined_collection.signal_validation_set.addOutputs( model.predict( [ self.__combined_collection.signal_validation_set.samples, self.__combined_collection.signal_validation_set.parameters ] ) )
            
            self.__combined_collection.background_training_set.addOutputs( model.predict( [ self.__combined_collection.background_training_set.samples, self.__combined_collection.background_training_set.parameters ] ) )
            self.__combined_collection.background_validation_set.addOutputs( model.predict( [ self.__combined_collection.background_validation_set.samples, self.__combined_collection.background_validation_set.parameters ] ) )
        
        elif self.__training_data_configuration['signal_parameters'] is not None:
            self.__combined_collection.signal_training_set.addOutputs( model.predict( self.__combined_collection.signal_training_set.samplesParametric ) )
            self.__combined_collection.signal_validation_set.addOutputs( model.predict( self.__combined_collection.signal_validation_set.samplesParametric ) )
            
            self.__combined_collection.background_training_set.addOutputs( model.predict( self.__combined_collection.background_training_set.samplesParametric ) )
            self.__combined_collection.background_validation_set.addOutputs( model.predict( self.__combined_collection.background_validation_set.samplesParametric ) )

        else :  
            self.__combined_collection.signal_training_set.addOutputs( model.predict( self.__combined_collection.signal_training_set.samples ) )
            self.__combined_collection.signal_validation_set.addOutputs( model.predict( self.__combined_collection.signal_validation_set.samples ) )
            
            self.__combined_collection.background_training_set.addOutputs( model.predict( self.__combined_collection.background_training_set.samples ) )
            self.__combined_collection.background_validation_set.addOutputs( model.predict( self.__combined_collection.background_validation_set.samples ) )

    
    #train gradient boosted forest in XGBoost
    @registerTrainingFunction( GradientBoostedForestConfiguration )
    def trainGradientBoostedForestClassificationModel( self, configuration ):

        training_set = self.__combined_collection.training_set()
        validation_set = self.__combined_collection.validation_set()

        feature_names = self.__training_data_configuration['list_of_branches']
        signal_parameters = self.__training_data_configuration['signal_parameters']
        is_parametric = ( signal_parameters is not None )
        if is_parametric:
            feature_names += signal_parameters
    
        trainGradientBoostedForestClassificationModel(
            train_data = ( training_set.samplesParametric if is_parametric else training_set.samples ),
            train_labels = training_set.labels, 
            train_weights = training_set.weights, 
            feature_names = feature_names,
            model_name = configuration.name(),
            number_of_trees = configuration['number_of_trees'],
            learning_rate = configuration['learning_rate'],
            max_depth = configuration['max_depth'],
            min_child_weight = configuration['min_child_weight'],
            subsample = configuration['subsample'],
            colsample_bytree = configuration['colsample_bytree'],
            gamma = configuration['gamma'],
            alpha = configuration['alpha'],
            number_of_threads = self.__training_data_configuration[ 'number_of_threads' ]
        )

        #load trained classifier 
        model = xgb.Booster()
        model.load_model( configuration.name() + '.bin' )

        #make xgboost DMatrices for predictions 
        signal_training_matrix = xgb.DMatrix( 
            self.__combined_collection.signal_training_set.samplesParametric if is_parametric else self.__combined_collection.signal_training_set.samples,
            label = self.__combined_collection.signal_training_set.labels, 
            nthread = self.__training_data_configuration[ 'number_of_threads' ]
        )
        signal_validation_matrix = xgb.DMatrix(
			self.__combined_collection.signal_validation_set.samplesParametric if is_parametric else self.__combined_collection.signal_validation_set.samples,
            label = self.__combined_collection.signal_validation_set.labels, 
            nthread = self.__training_data_configuration[ 'number_of_threads' ]
        )

        background_training_matrix = xgb.DMatrix(
            self.__combined_collection.background_training_set.samplesParametric if is_parametric else self.__combined_collection.background_training_set.samples,
            label = self.__combined_collection.background_training_set.labels, 
            nthread = self.__training_data_configuration[ 'number_of_threads' ]
        )
        background_validation_matrix = xgb.DMatrix(
			self.__combined_collection.background_validation_set.samplesParametric if is_parametric else self.__combined_collection.background_validation_set.samples,
            label = self.__combined_collection.background_validation_set.labels, 
            nthread = self.__training_data_configuration[ 'number_of_threads' ]
        )
        
        #make predictions 
        self.__combined_collection.signal_training_set.addOutputs( model.predict( signal_training_matrix ) )
        self.__combined_collection.signal_validation_set.addOutputs( model.predict( signal_validation_matrix ) )

        self.__combined_collection.background_training_set.addOutputs( model.predict( background_training_matrix ) )
        self.__combined_collection.background_validation_set.addOutputs( model.predict( background_validation_matrix ) )


#some testing code 

if __name__ == '__main__' :
    configuration_file = __import__( 'input_ewkino' )
    reader = TrainingDataReader( configuration_file )
    setup = ModelTrainingSetup( reader )

    #neural network
    config_dict = { 'batch_size': 1024,
        'dropout_all' : True,
        'dropout_first' : False,
        'dropout_rate' : 0.5,
        'learning_rate' : 1,
        'learning_rate_decay' : 1,
        'number_of_hidden_layers' : 5,
        'number_of_epochs' : 1,
        'optimizer' : 'Nadam',
        'units_per_layer': 128,
        'activation' : 'prelu',
        'batchnorm_first' : True,
        'batchnorm_hidden' : True,
        'batchnorm_before_activation' : True
    }
    config = newConfigurationFromDict( **config_dict )
    setup.trainAndEvaluateModel( config )

    #gradient boosted forest
    config_dict = { 'alpha' : 0,
        'colsample_bytree' : 1,
        'gamma' : 0,
        'learning_rate' : 0.05,
        'max_depth' : 3,
        'min_child_weight' : 1,
        'number_of_trees' : 100,
        'subsample' : 0.5
    }

    config = newConfigurationFromDict( **config_dict )
    setup.trainAndEvaluateModel( config )
