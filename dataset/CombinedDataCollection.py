import numpy as np

#import other parts of framework
import os
import sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )
from dataset.Dataset import concatenateAndShuffleDatasets
from parametrization.ParameterGenerator import ParameterGenerator


#def _numberOfBatches( dataset, batch_size ):
#    return ( int( len( dataset )/batch_size ) + bool( len( dataset )%batch_size ) )
def _numberOfBatches( dataset_length, batch_size ):
    return ( int( dataset_length / batch_size ) + bool( dataset_length % batch_size ) )


def _combineDataset( signal_dataset, background_dataset ):
    if signal_dataset.isParametric():
        parameter_generator = ParameterGenerator( signal_dataset.parameters )
        background_dataset.addParameters( parameter_generator.yieldRandomParameters( len( background_dataset ) ) )
    return concatenateAndShuffleDatasets( signal_dataset, background_dataset )


def _dataGenerator( signal_dataset, background_dataset, batch_size ):
    is_parametric = signal_dataset.isParametric()
    if is_parametric:
        parameter_generator = ParameterGenerator( signal_dataset.parameters )
    number_of_batches_per_epoch = _numberOfBatches( len( signal_dataset ) + len( background_dataset ), batch_size )
    while True:
        if is_parametric:
            background_parameters = parameter_generator.yieldRandomParameters( len( background_dataset ) )
            background_dataset.addParameters( background_parameters )
        combined_dataset = concatenateAndShuffleDatasets( signal_dataset, background_dataset )

        for batch_index in range( number_of_batches_per_epoch ):
            batch = combined_dataset[ batch_index*batch_size : ( batch_index + 1 )*batch_size ]
            if is_parametric:
                yield ( batch.samplesParametric, batch.labels, batch.weights )
            else:
                yield ( batch.samples, batch.labels, batch.weights )



class CombinedDataCollection :
    
    def __init__( self, signal_collection, background_collection ):
        self.__signal_collection = signal_collection
        self.__background_collection = background_collection


    @property
    def signal_collection( self ):
        return self.__signal_collection
    
        
    @property 
    def background_collection( self ):
        return self.__background_collection


    @property
    def signal_training_set( self ):
        return self.signal_collection.training_set

    
    @property 
    def signal_validation_set( self ):
        return self.signal_collection.validation_set


    @property
    def signal_test_set( self ):
        return self.signal_collection.test_set 

    
    @property
    def background_training_set( self ):
        return self.background_collection.training_set 

        
    @property
    def background_validation_set( self ):
        return self.background_collection.validation_set 


    @property
    def background_test_set( self ):
        return self.background_collection.test_set 


    def __getDataset( self, attribute_name, collection_attribute_name ):
        try:
            return getattr( self, attribute_name )
        except AttributeError:
            combined_dataset = _combineDataset( getattr( self.__signal_collection, collection_attribute_name ), getattr( self.__background_collection, collection_attribute_name ) )
            setattr( self, attribute_name, combined_dataset )
            return combined_dataset 


    def training_set( self ):
        return self.__getDataset( '__training_set', 'training_set' )


    def validation_set( self ):
        return self.__getDataset( '__validation_set', 'validation_set' )


    def test_set( self ):
        return self.__getDataset( '__test_set', 'test_set' )

        
    def numberOfTrainingBatches( self, batch_size ):
        combined_length = len( self.__signal_collection.training_set ) + len( self.__background_collection.training_set )
        return _numberOfBatches( combined_length, batch_size )
    

    def trainingGenerator( self, batch_size ):
        return _dataGenerator( self.__signal_collection.training_set, self.__background_collection.training_set, batch_size )
        

    def numberOfValidationBatches( self, batch_size ):
        combined_length = len( self.__signal_collection.validation_set ) + len( self.__background_collection.validation_set )
        return _numberOfBatches( combined_length, batch_size )


    def validationGenerator( self, batch_size ):
        return _dataGenerator( self.__signal_collection.validation_set, self.__background_collection.validation_set, batch_size )


    def numberOfTestBatches( self, batch_size ):
        combined_length = len( self.__signal_collection.validation_set ) + len( self.__background_collection.validation_set )
        return combined_length


    def testGenerator( self, batch_size ):
        return _dataGenerator( self.__signal_collection.test_set, self.__background_collection.test_set, batch_size )
