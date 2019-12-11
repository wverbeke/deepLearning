"""
Class that collects a numpy array of features, and the corresponding weight for every event 
"""

#import python libraries
import numpy as np


#return an array of randomly shuffled indices for a given array
def randomlyShuffledIndices( array ):
    return np.random.permutation( len( array ) )



#class collecting a set of samples, together with the weights and labels for each sample
class Dataset:

    def __init__( self, samples, weights, labels, outputs = None, parameters = None ):

        #make sure each sample has a weight and label and vice-versa
        if len(samples) != len(weights) or len(samples) != len(labels):
            raise IndexError( 'sample, weight and label arrays must have equal length!' )

        self.__samples = samples 
        self.__weights = weights  
        self.__labels = labels
        if outputs is not None:
            self.__outputs = outputs

        if parameters is not None:
            self.__parameters = parameters


    @property
    def samples( self ):
        return self.__samples
 
   
    @property
    def weights( self ):
        return self.__weights


    @property
    def labels( self ):
        return self.__labels


    @property
    def outputs( self ):
        try:
            return self.__outputs
        except AttributeError:
            return None

    
    @property 
    def parameters( self ):
        try:
            return self.__parameters
        except AttributeError:
            return None


    @property
    def samplesParametric( self ):
        try:
            return np.concatenate( [ self.samples, self.parameters ], axis = -1 )
        except ValueError:
            return self.samples


    def __len__( self ):
        return len( self.samples )


    def addOutputs( self, outputs ):
        if len( outputs ) != len( self ):
            raise IndexError( 'outputs must have the same length as the Dataset object.' )
        self.__outputs = outputs


    def addParameters( self, parameters ):
        if len( parameters ) != len( self ):
            raise IndexError( 'parameters must have the same length as the Dataset object.' )
        self.__parameters = parameters


    def __add__( self, rhs ):
        samples = np.concatenate( (self.samples, rhs.samples), axis = 0)
        weights = np.concatenate( (self.weights, rhs.weights), axis = 0)
        labels = np.concatenate( (self.labels, rhs.labels), axis = 0)

        outputs = None
        try:
            outputs = np.concatenate( ( self.outputs, rhs.outputs ), axis = 0)
        except ValueError:
            pass

        parameters = None
        try:
            parameters = np.concatenate( ( self.parameters, rhs.parameters ), axis = 0)
        except ValueError:
            pass

        return Dataset( samples, weights, labels, outputs, parameters )


    def __getitem__( self, index ):
        samples = self.__samples[index]
        weights = self.__weights[index]
        labels = self.__labels[index]
        try:
            outputs = self.__outputs[index]
        except AttributeError:
            outputs = None
        try:
            parameters = self.__parameters[index]
        except AttributeError:
            parameters = None
        return Dataset( samples, weights, labels, outputs, parameters )


    def shuffle( self ):
        shuffled_indices = randomlyShuffledIndices( self )
        self.__samples = self.__samples[ shuffled_indices ]
        self.__weights = self.__weights[ shuffled_indices ]
        self.__labels = self.__labels[ shuffled_indices ]
        try:
            self.__outputs = self.__outputs[ shuffled_indices ]
        except AttributeError:
            pass
        try:
            self.__parameters = self.__parameters[ shuffled_indices ]
        except AttributeError:
            pass

        
    #allow scaling of weights
    def scaleWeights( self, scale_factor ):
        self.__weights *= scale_factor


    def isParametric( self ):
        return ( self.parameters is not None )
    


#add two Datasets and shuffle them afterwards
def concatenateAndShuffleDatasets( lhs_dataset, rhs_dataset):
    merged_set = lhs_dataset + rhs_dataset 
    merged_set.shuffle()
    return merged_set 
