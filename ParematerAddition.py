'''
This is a helper script for doing a parametrized model training for new physics searches
For explanation of the method see https://arxiv.org/abs/1601.07913
Read a parameter from the signal (such as a new physics particle mass) from a signal tree
Fill the background tree with randomly samples parameters from the total signal distribution 
'''

from treeToArray import treeToArray
import numpy as np

from ROOT import TFile, TTree, TBranch
import sys

class ParameterAdder():
    def __init__(self, signal_tree, parameter_name):
        self.parameter_name = parameter_name 
        total_count = 0
        parameter_counts = {} 
        parameter_array = treeToArray(signal_tree, parameter_name, cut = '')
        for entry in parameter_array:
            if entry in parameter_counts:
                parameter_counts[entry] += 1
            else :
                parameter_counts[entry] = 1
            total_count += 1
        
        self.parameters = []
        self.probabilities = []
        for parameter in parameter_counts:
            self.parameters.append( parameter )
            self.probabilities.append( float(parameter_counts[parameter]) / total_count ) 
         

    #generate one or multiple random parameters 
    def yieldRandomParameters(self, size = None):
        return np.random.choice( self.parameters, p = self.probabilities, size = size)

    def updateTree(self, tree_to_update):
        num_events = tree_to_update.GetEntries()
        random_parameters = self.yieldRandomParameters( num_events )
        
        parameter_to_fill = np.array(0., dtype=np.float32)
        parameter_branch = tree_to_update.Branch(self.parameter_name, parameter_to_fill , '{}/F'.format( self.parameter_name) )
        for i, event in enumerate(tree_to_update):
            tree_to_update.GetEntry( i )
            parameter_to_fill = np.array(random_parameters[i], dtype=np.float32)
            parameter_branch.Fill()
        tree_to_update.Write()

	
if __name__ == '__main__' :
    
    if len( sys.argv ) != 5:
        print('Incorrect number of command line argument given. Aborting.')
        print('Usage : <python ParematerAddition.py root_file_name signal_tree_name background_tree_name parameter_name >')
        sys.exit()

    else:

        # read trees from root file
        root_file = TFile(sys.argv[1], 'update')
        signal_tree = root_file.Get( sys.argv[2] )
        background_tree = root_file.Get( sys.argv[3] )
        
        #make ParameterAdder and use it to add signal parameter to background tree
        adder = ParameterAdder(signal_tree, sys.argv[4])
        adder.updateTree( background_tree )
