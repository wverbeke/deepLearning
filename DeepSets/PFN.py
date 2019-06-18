"""
Class that represents a particle flow network, as explained in https://arxiv.org/abs/1810.05165
"""

from keras import Input, Model, callbacks, layers, backend
import tensorflow as tf

#include other parts of framework
import os
import sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert(0, main_directory )
from configuration.Optimizer import Optimizer
from diagnosticPlotting import plotKerasMetricComparison


#custom AUC metric 
def auc( true_labels, predictions ):
    auc = tf.metrics.auc( true_labels,  predictions )[1]
    backend.get_session().run( tf.local_variables_initializer() )
    return auc


#particle flow network
class PFN:
    
    def __init__( self, input_shape_particleFlow, input_shape_highlevel, num_hidden_layers_latent = 2, nodes_per_layer_latent = 128, batch_normalization_latent = True, dropout_rate_latent = 0, latent_space_size = 256, num_hidden_layers_output = 2, nodes_per_layer_output = 128, batch_normalization_output = True, dropout_rate_output = 0, optimizer_name = 'Nadam', relative_learning_rate = 1, relative_learning_rate_decay = 1):
    
        self.__input_shape_particleFlow = input_shape_particleFlow
        self.__input_shape_highlevel = input_shape_highlevel
        
        self.__num_hidden_layers_latent  = num_hidden_layers_latent
        self.__nodes_per_layer_latent = nodes_per_layer_latent
        self.__batch_normalization_latent = batch_normalization_latent
        self.__dropout_rate_latent = dropout_rate_latent
        self.__latent_space_size = latent_space_size 
        
        
        self.__num_hidden_layers_output = num_hidden_layers_output
        self.__nodes_per_layer_output = nodes_per_layer_output
        self.__batch_normalization_output = batch_normalization_output
        self.__dropout_rate_output = dropout_rate_output
        
        self.__model = self.__produceModel() 
        self.__model.summary()
        self.__optimizer = Optimizer( optimizer_name, relative_learning_rate, relative_learning_rate_decay )
        self.__compileModel()
        

    def __produceModel( self ):

        #particle flow candidate inputs 
        particleFlow_input = Input( shape = self.__input_shape_particleFlow )
        
        #Set up neural network that transforms the representation of each particle to a latent space 

        #apply masking to 0-valued particles
        particleFlow_intermediate = layers.Masking( mask_value = 0 )( particleFlow_input )
        for l in range( self.__num_hidden_layers_latent ):
        	if self.__batch_normalization_latent :
        		particleFlow_intermediate = layers.BatchNormalization()( particleFlow_intermediate )
        
        	if self.__dropout_rate_latent > 0 :
        		particleFlow_intermediate = layers.TimeDistributed( layers.Dropout( self.__dropout_rate_latent ) )( particleFlow_intermediate )
        
        	particleFlow_intermediate = layers.TimeDistributed( layers.Dense( self.__nodes_per_layer_latent, activation = 'relu' ) )( particleFlow_intermediate )
        
        if self.__batch_normalization_latent:
        	particleFlow_intermediate = layers.BatchNormalization()( particleFlow_intermediate )
        
        if self.__dropout_rate_latent > 0 :
        	particleFlow_intermediate = layers.TimeDistributed( layers.Dropout( self.__dropout_rate_latent ) )( particleFlow_intermediate )
        
        particleFlow_intermediate = layers.TimeDistributed( layers.Dense( self.__latent_space_size, activation = 'linear' ) )( particleFlow_intermediate )
        
        #apply summation in latent space
        def summation( x ):
        	x = backend.sum( x, axis = 1 )
        	return x
        particleFlow_intermediate = layers.Lambda( summation, output_shape=None, mask=None, arguments=None)( particleFlow_intermediate )
        
        #high level inputs to be added on top of the particle representations
        highlevel_input = Input( shape = self.__input_shape_highlevel )
        
        #connect high level inputs to latent space representation of jet
        merged_intermediate = layers.concatenate( [ particleFlow_intermediate, highlevel_input ], axis = -1 )
        
        #neural network to go from latent space representation and high level inputs to solution
        for l in range( self.__num_hidden_layers_output ):
            if self.__batch_normalization_output :
                merged_intermediate = layers.BatchNormalization()( merged_intermediate )

            if self.__dropout_rate_output > 0 :
                merged_intermediate = layers.Dropout( self.__dropout_rate_output )( merged_intermediate )
        
            merged_intermediate = layers.Dense( self.__nodes_per_layer_output, activation = 'relu' )( merged_intermediate )
        
        output = layers.Dense( 1, activation = 'sigmoid' )( merged_intermediate )
        
        #build the model
        model = Model( [particleFlow_input, highlevel_input] , output )
        return model


    def __compileModel( self ):
        self.__model.compile(
            optimizer = self.__optimizer.kerasOptimizer(),
            loss = 'binary_crossentropy',
            metrics = [ auc, 'acc']
        )

    
    def trainModel( self, sample_generator, output_name, batch_size = 512, number_of_epochs = 100, patience = 4):
        callback_list = [
            callbacks.EarlyStopping(
                monitor = 'auc',
                mode = 'max',
                patience = patience
            ),
            callbacks.ModelCheckpoint(
                monitor = 'val_auc',
                mode = 'max',
                filepath = output_name,
                save_best_only = True
            )
        ]
        
        history = self.__model.fit_generator(
        	sample_generator.trainingGenerator( batch_size ),
        	epochs = number_of_epochs,
        	steps_per_epoch = sample_generator.numberOfTrainingBatches( batch_size ),
        	validation_data = sample_generator.validationGenerator( batch_size ),
        	validation_steps = sample_generator.numberOfValidationBatches( batch_size ),
        	callbacks = callback_list
        )
        
        plotKerasMetricComparison( history, 'jetTagger', 'auc', 'AUC' )
        plotKerasMetricComparison( history, 'jetTagger', 'acc', 'Accuracy' )
        plotKerasMetricComparison( history, 'jetTagger', 'loss', 'Loss' )



if __name__ == '__main__' :
    pass
