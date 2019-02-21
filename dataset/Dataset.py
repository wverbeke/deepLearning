"""
Class that collects a numpy array of features, and the corresponding weight for every event 
"""

#import python libraries
import numpy as np


#return an array of randomly shuffled indices for a given array
def randomlyShuffledIndices( array ):
    indices = np.arange( len(array) )
    np.random.shuffle( indices )
    return indices 



#class collecting a set of samples, together with the weights and labels for each sample
class Dataset:
    def __init__( self, samples, weights, labels ):

        #make sure each sample has a weight and label and vice-versa
        if len(samples) != len(weights) or len(samples) != len(labels):
            raise IndexError( 'sample, weight and label arrays must have equal length!' )

        self.__samples = samples 
        self.__weights = weights  
        self.__labels = labels

    @property
    def samples( self ):
        return self.__samples
    
    @property
    def weights( self ):
        return self.__weights

    @property
    def labels( self ):
        return self.__labels

    def __len__( self ):
        return len( self.samples )

    def __add__( self, rhs ):
        samples = np.concatenate( (self.samples, rhs.samples), axis = 0)
        weights = np.concatenate( (self.weights, rhs.weights), axis = 0)
        labels = np.concatenate( (self.labels, rhs.labels), axis = 0)
        return Dataset(samples, weights, labels)

    def __getitem__( self, index ):
        return Dataset( self.__samples[index], self.__weights[index], self.__labels[index] )

    def shuffle( self ):
        shuffled_indices = randomlyShuffledIndices( self )
        self.__samples = self.__samples[ shuffled_indices ]
        self.__weights = self.__weights[ shuffled_indices ]
        self.__labels = self.__labels[ shuffled_indices ]



#add two Datasets and shuffle them afterwards
def concatenateAndShuffleSets( lhs_dataset, rhs_dataset):
    merged_set = lhs_dataset + rhs_dataset 
    merged_set.shuffle()
    return merged_set 
