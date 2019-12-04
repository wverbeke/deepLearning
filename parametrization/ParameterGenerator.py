'''
This is a helper script for doing a parametrized model training for new physics searches
For explanation of the method see https://arxiv.org/abs/1601.07913
Given an array of signal parameters, generate random parameter arrays from the same distribution
'''

import numpy as np
import sys
import collections
import array
from ROOT import TFile, TTree, TBranch

#import other parts of framework
import os
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )
from treeToArray import treeToArray



class ParameterGenerator():

    def __init__( self, parameter_array ):
        self.__parameter_array = parameter_array
        

    #yield random parameter array
    def yieldRandomParameters( self, size = None ):
        ret_array = None
        current_size = size
        input_size = len( self.__parameter_array )
        while current_size >= input_size :
            np.random.shuffle( self.__parameter_array )
            if ret_array is None:
                ret_array = self.__parameter_array
            else:
                ret_array = np.concatenate( [ ret_array, self.__parameter_array ], axis = 0 )
            current_size -= input_size
        if current_size < input_size and current_size != 0:
            np.random.shuffle( self.__parameter_array )
            if ret_array is None:
                ret_array = self.__parameter_array[0:size]
            else:
                ret_array = np.concatenate( [ ret_array, self.__parameter_array ], axis = 0 )
        return ret_array


    #yield a single random parameter by walking through the shuffled parameter array
    def __parameterGenerator( self ):
        while True:
            np.random.shuffle( self.__parameter_array )
            for entry in self.__parameter_array:
                yield entry

    __generator = None
    def yieldRandomParameter( self ):
        if self.__generator is None :
            self.__generator = self.__parameterGenerator()
        return next( self.__generator )
            

class ParameterGenerator_memoryEfficient():

    def __init__( self, parameter_array ):
        self.__is_multi_parameter = ( len( parameter_array.shape ) > 1 )
        if self.__is_multi_parameter :
            parameters = [ tuple( entry ) for entry in parameter_array ]
        else:
            parameters = parameter_array

        #counts of each parameter
        parameter_counts = collections.Counter( parameters )
        total_count = len( parameter_array )

        #compute probabilities for each parameter combination
        self.__parameter_options = []
        self.__probabilities = []
        for parameter_option, count in parameter_counts.items():
            self.__parameter_options.append( parameter_option )
            self.__probabilities.append( float( count ) / total_count )

        self.__parameter_options = np.array( self.__parameter_options )

    
    def yieldRandomParameters( self, size = None ):
        if self.__is_multi_parameter:

            #np.random.choice only works on 1D arrays so instead choose from a 1D list of indices, which is then used to index the parameter array to simultaneously yield n parameters
            possible_indices = np.arange( len( self.__parameter_options ) )
            random_indices = np.random.choice( possible_indices, p = self.__probabilities, size = size )
            return self.__parameter_options[ random_indices ]

        else:
            return np.random.choice( self.__parameter_options, p = self.__probabilities, size = size )


    def yieldRandomParameter( self ):
        return self.yieldRandomParameters( 1 )[0]



#some testing code 
if __name__ == '__main__':
    test_params_2D_p1 = np.random.randn( 100 ) 
    test_params_2D_p2 = np.random.randn( 100 )
    test_params_2D_p1 = np.expand_dims( test_params_2D_p1, axis = 1 )
    test_params_2D_p2 = np.expand_dims( test_params_2D_p2, axis = 1 )
    test_params_2D = np.concatenate( [test_params_2D_p1, test_params_2D_p2], axis = 1 )
    parGen = ParameterGenerator( test_params_2D )
    random_params = parGen.yieldRandomParameters( 10 )
    
    import matplotlib.pyplot as plt
    plt.subplot( 1, 2, 1 )
    plt.scatter( test_params_2D[:,0], test_params_2D[:,1] )
    plt.xlim( [-3,3] )
    plt.ylim( [-3,3] )
    plt.subplot( 1, 2, 2 )
    plt.scatter( random_params[:, 0], random_params[:,1] )
    plt.xlim( [-3,3] )
    plt.ylim( [-3,3] )
    plt.savefig('parametrization.pdf')
