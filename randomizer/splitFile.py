import uproot
from ROOT import TTree, TFile
import os
import sys

#import other parts of framework
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
print( main_directory )
sys.path.insert( 0, main_directory )
from randomizer.BranchInfo import BranchInfoCollection
from randomizer.eventSize import numberOfEventsToRead
from miscTools.stringTools import removePathAndExtension


#return set with all possible values for given branch
def possibleValues( uproot_branch ):
    possible_values = set()
    for value in uproot_branch.array():
        possible_values.add( value )
    return possible_values


def splitFile( input_file_name, input_tree_name, splitting_branch_name ):
    input_file = uproot.open( input_file_name )
    input_tree = input_file[ input_tree_name ]
    
    branch_collection = BranchInfoCollection( input_tree ) 
    
    #determine possible values of branch according to which the file will be split
    splitting_branch = input_tree[ splitting_branch_name ] 
    split_values = possibleValues( splitting_branch )

    #make new file with split trees 
    split_file_name = '{}_split_{}.root'.format( removePathAndExtension( input_file_name ), splitting_branch_name )
    split_file = TFile( split_file_name, 'RECREATE' )
    
    split_trees = {}
    for value in split_values:
        name = '{}_{}_{}'.format( input_tree_name, splitting_branch_name, value )
        split_trees[value] = TTree( name, name )
        branch_collection.addBranches( split_trees[value] )
        
    #loop over the original tree and write the events to the corresponding new trees 
    events_per_iteration = numberOfEventsToRead( input_tree )
    for i in range( 0, len(input_tree), events_per_iteration ):
        loaded_arrays = { key.decode('utf-8') : value for key, value in input_tree.arrays( entrystart = i, entrystop = i + events_per_iteration ).items() }
        split_array = loaded_arrays[ splitting_branch_name ]
        size = min( events_per_iteration, len( list( loaded_arrays.values() )[0] ) )
        for j in range( size ):
            branch_collection.fillArrays( loaded_arrays, j )
            split_trees[ split_array[j] ].Fill()

    for tree in split_trees.values():
        tree.Write()
    split_file.Close()


if __name__ == '__main__':
    if len( sys.argv ) == 4:
        input_file_name = sys.argv[1] 
        input_tree_name = sys.argv[2] 
        splitting_branch_name = sys.argv[3]
        splitFile( input_file_name, input_tree_name, splitting_branch_name )
    else:
        print('Error: invalid number of arguments given.' )
        print('Usage: python splitFile.py <input_file_name> <input_tree_name> <splitting_branch_name>') 
 
