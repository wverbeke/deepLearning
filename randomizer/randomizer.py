import uproot
import numpy as np

import time
import sys 
import os
from ROOT import TFile, TTree

#import other parts of framework
import sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )
from randomizer.BranchInfo import BranchInfo, BranchInfoCollection
from randomizer.eventSize import numberOfFileSplittings, numberOfEventsToRead
from randomizer.shuffle import randomIndices
from randomizer.TreeIterator import TreeIterator


def splitFileName( original_path ):
    file_name = os.path.basename( original_path )
    return ( '{}_randomSplit.root'.format( file_name.split('.')[0] ) )


def randomizedFileName( original_path ):
    file_name = os.path.basename( original_path )
    return ( '{}_randomized.root'.format( file_name.split('.')[0] ) )


def getFileKeys( root_file ):
    seen_keys = set()
    for key in root_file.keys():
        key = key.decode('utf-8')
        key = key.split(';')[0]
        if key in seen_keys:
            continue
        yield key
        seen_keys.add( key )


def createRandomizedFile( input_file_name ):

    f = uproot.open( input_file_name )
    for input_tree_name in getFileKeys( f ):
        
        tree = f[input_tree_name]
        
        #collection of branches in input file and their information
        branch_collection = BranchInfoCollection( tree ) 

        #split root file randomly into root_file    
        number_of_splittings = numberOfFileSplittings( tree )
        split_file = TFile( splitFileName( input_file_name ), 'RECREATE' )
        split_trees = [ ]
        for i in range( number_of_splittings ):
            t = TTree( input_tree_name + str(i), input_tree_name + str(i) )
            branch_collection.addBranches( t )
            split_trees.append( t )
        
        for array_map in TreeIterator( tree ):

            #write each event to a random input file 
            random_file_choices = np.random.choice( np.arange( number_of_splittings ), len( array_map ) )
            for j in range( len( array_map ) ):
                branch_collection.fillArrays( array_map, j )
                split_trees[ random_file_choices[j] ].Fill() 
        
        for t in split_trees:
            t.Write()
        split_file.Close()
        
        #read in new root file 
        split_file = uproot.open( splitFileName( input_file_name ) )
        split_trees = [ split_file[key] for key in getFileKeys( split_file ) ]
        
        #make randomized output file to which to write the split trees after shuffling 
        randomized_file = TFile( randomizedFileName( input_file_name ), 'RECREATE')
        randomized_tree = TTree( input_tree_name, input_tree_name )
        branch_collection.addBranches( randomized_tree )
        
        for t in split_trees:
        
            #load arrays for each of the split trees 
            loaded_arrays = { key.decode('utf-8') : value for key, value in t.arrays().items() }
        
            #randomly shuffle all arrays
            size = len( t )
            random_indices = randomIndices( size )
        
            for key in loaded_arrays:
                loaded_arrays[key] = loaded_arrays[key][random_indices]
        
            for j in range( size ):
                branch_collection.fillArrays( loaded_arrays, j )
                randomized_tree.Fill()
        
        #write fully randomized file        
        randomized_tree.Write()
        randomized_file.Close()

        #clean up temporary split file
        os.remove( splitFileName( input_file_name ) )
        


if __name__ == '__main__':
    
    if len( sys.argv ) == 2 :
        input_file_name = sys.argv[1]
        createRandomizedFile( input_file_name )

    else :
        print('Error: invalid number of arguments given.' )
        print('Usage: python randomizer.py <input_file_name>')
