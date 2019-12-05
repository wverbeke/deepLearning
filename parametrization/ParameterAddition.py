'''
This is a helper script for doing a parametrized model training for new physics searches
For explanation of the method see https://arxiv.org/abs/1601.07913
Read a parameter from the signal (such as a new physics particle mass) from a signal tree
Fill the background tree with randomly samples parameters from the total signal distribution 
'''

import numpy as np
import array
from ROOT import TFile, TTree, TBranch
from collections import OrderedDict

#import other parts of framework
import os, sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )
from parametrization.ParameterGenerator import ParameterGenerator, ParameterGenerator_memoryEfficient
from treeToArray import treeToArray



class ParameterAdder():

    def __init__( self, signal_tree, parameter_names ):

        #make sure it works with either a list of parameters or a single string representing one parameter
        if isinstance( parameter_names, str ):
            self.__parameter_names = []
            self.__parameter_names.append( parameter_names )
        else :
            self.__parameter_names = list( parameter_names )
            
        parameter_array = treeToArray( signal_tree, self.__parameter_names, cut = '' )
        self.__parameter_generator = ParameterGenerator( parameter_array )


    #add parameters to another tree
    def parametrizeTree( self, tree_to_update ):

        #yield random set of parameters for each event
        number_of_events = tree_to_update.GetEntries()

        #make branches and variables for filling the new parameters 
        values_to_fill = [ array.array( 'f', [0] ) for parameter in self.__parameter_names ]
        parameter_branches = [ tree_to_update.Branch( parameter, value_to_fill, parameter + '/F' ) for parameter, value_to_fill in zip( self.__parameter_names, values_to_fill ) ]

        #loop over tree and write the new parameter values 
        for i, _ in enumerate( tree_to_update ):
            tree_to_update.GetEntry( i )
            random_parameter_combination = self.__parameter_generator.yieldRandomParameter()
            for i, parameter in enumerate( random_parameter_combination ):
                values_to_fill[ i ][ 0 ] = parameter
            for branch in parameter_branches:
                branch.Fill()
        tree_to_update.Write()



class ParameterAdderSingleTree():
    
    def __init__( self, tree, parameter_names, background_defaults, is_uproot = False ):

        try:
            if len( parameter_names ) != len( background_defaults ):
                raise IndexError( 'Number of parameter names and number of background_defaults must be equal!' )
        except:
            pass
            

        if isinstance( parameter_names, str ):
            self.__parameter_names = [] 
            self.__parameter_names.append( parameter_names ) 
        else:
            self.__parameter_names = list( parameter_names )
    
        try:        
            self.__background_defaults = list( background_defaults )
        except TypeError:
            self.__background_defaults = []
            self.__background_defaults.append( background_defaults )

        #make array of signal parameters
        if not is_uproot :
            full_parameter_array = treeToArray( tree, self.__parameter_names, cut = '' )
        else :
            full_parameter_array = np.concatenate( [ np.expand_dims( tree.array( key ), axis = 1 ) for key in self.__parameter_names ], axis = 1 )
            
        signal_parameter_list = []
        for entry in full_parameter_array :
            is_background = True
            for i, sub_entry in enumerate( entry ):

                #make this robuster in the future by avoiding direct floating point comparison
                if sub_entry != self.__background_defaults[ i ]:
                    is_background = False
            if not is_background:
                signal_parameter_list.append( entry )

        #signal_parameter_array = full_parameter_array[ full_parameter_array != background_default ]
        signal_parameter_array = np.array( signal_parameter_list )

        #make parameter generator from signal parameter array
        self.__parameter_generator = ParameterGenerator( signal_parameter_array )


    def parametrizeTree( self, tree_to_update ):
        parameters_to_fill = [ array.array( 'f', [0] ) for parameter in self.__parameter_names ]
        parameter_branches = [ tree_to_update.Branch( parameter + '_parametrized', parameter_to_fill, parameter + '_parametrized/F' ) for parameter, parameter_to_fill in zip( self.__parameter_names, parameters_to_fill ) ]

        for i, _ in enumerate( tree_to_update ):
            tree_to_update.GetEntry( i )
            #current_parameter_value =  getattr( tree_to_update, self.__parameter_name )
            current_parameter_values = [ getattr( tree_to_update, parameter ) for parameter in self.__parameter_names ]
            
            if current_parameter_values == self.__background_defaults:

                #for background randomly sample parameter from signal distribution
                for i, entry in enumerate( self.__parameter_generator.yieldRandomParameter() ):
                    parameters_to_fill[ i ][ 0 ] = entry 
                
            else:

                #for signal retain current parameter
                for i, entry in enumerate( current_parameter_values ):
                    parameters_to_fill[ i ][ 0 ] = entry 

            for branch in parameter_branches:
                branch.Fill()
        tree_to_update.Write()


    def yieldRandomParameter( self ):
        return self.__parameter_generator.yieldRandomParameter()


	
if __name__ == '__main__' :
    pass
