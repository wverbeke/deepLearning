'''
This is a helper script for doing a parametrized model training for new physics searches
For explanation of the method see https://arxiv.org/abs/1601.07913
Read a parameter from the signal (such as a new physics particle mass) from a signal tree
Fill the background tree with randomly samples parameters from the total signal distribution 
'''

import numpy as np
import sys
import collections
from treeToArray import treeToArray
from ROOT import TFile, TTree, TBranch



class ParameterAdder():

    def __init__( self, signal_tree, parameter_name ):
        self._parameter_name = parameter_name 

        parameter_array = treeToArray( signal_tree, parameter_name, cut = '' )
        parameter_counts = collections.Counter( parameter_array )
        total_count = len( parameter_array )
        
        self._parameters = []
        self._probabilities = []
        for parameter in parameter_counts:
            self._parameters.append( parameter )
            self._probabilities.append( float( parameter_counts[parameter] ) / total_count ) 
         

    #generate one or multiple random parameters 
    def _yieldRandomParameters( self, size = None ):
        return np.random.choice( self._parameters, p = self._probabilities, size = size )


    #add parameter tree
    def updateTree( self, tree_to_update ):
        num_events = tree_to_update.GetEntries()
        random_parameters = self._yieldRandomParameters( num_events )
        
        parameter_to_fill = np.array( 0., dtype = np.float32 )
        parameter_branch = tree_to_update.Branch( self._parameter_name, parameter_to_fill , '{}/F'.format( self._parameter_name) )
        for i, event in enumerate( tree_to_update ):
            tree_to_update.GetEntry( i )
            parameter_to_fill = np.array( random_parameters[i], dtype = np.float32 )
            parameter_branch.Fill()
        tree_to_update.Write()


	
if __name__ == '__main__' :
    
    if not( len( sys.argv ) == 5  or len( sys.argv ) == 6 ):
        print('Incorrect number of command line argument given. Aborting.')
        print('Usage : <python ParematerAddition.py root_file_name signal_tree_name background_tree_name parameter_name >')
        print('Or alternatively: <python ParematerAddition.py signal_root_file_name signal_tree_name background_root_file_name background_tree_name parameter_name >')
        sys.exit()

    else:
        
        # read trees from root file
        # signal and background are in the same file 
        if len( sys.argv ) == 5:
            root_file = TFile( sys.argv[1], 'update' )
            signal_tree = root_file.Get( sys.argv[2] )
            background_tree = root_file.Get( sys.argv[3] )
        
        #signal and background are in a different file
        else:
            signal_root_file = TFile( sys.argv[1] )
            signal_tree = signal_root_file.Get( sys.argv[2] )
            background_root_file = TFile( sys.argv[3], 'update' )
            background_tree = background_root_file.Get( sys.argv[4] )
        
        #make ParameterAdder and use it to add signal parameter to background tree
        parameter_name = sys.argv[4] if len( sys.argv ) == 5 else sys.argv[5]
        adder = ParameterAdder( signal_tree, parameter_name )
        adder.updateTree( background_tree )
