import uproot

#import other parts of framework
import os
import sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert(0, main_directory )
from DeepSets.PFN import PFN
from DeepSets.PFNSampleGenerator import PFNSampleGenerator



if __name__ == '__main__':
    
    #read root file and tree using uproot
    #fill in your file and tree names here
    root_file_name = '~/Work/jetTagger/SampleGenerator/mergedFile_randomized.root'
    tree_name = 'HNLtagger_tree'

    f = uproot.open( root_file_name )
    tree = f[ tree_name ]
    
    #make a Sample generator 
    #fill in the names of your particle-flow and high level branch names here
    pfn_branch_names = [ '_JetConstituentPt', '_JetConstituentEta', '_JetConstituentPhi', '_JetConstituentdxySig', '_JetConstituentsNumberOfHits', '_JetConstituentsNumberOfPixelHits', '_JetConstituentCharge', '_JetConstituentPdgId', '_JetConstituentdzSig']
    highlevel_branch_names = [ '_JetPt', '_JetEta' ]
    label_branch = '_JetIsFromHNL'
    
    sample = PFNSampleGenerator( tree, pfn_branch_names, highlevel_branch_names, label_branch, validation_fraction = 0.4, test_fraction = 0.2 )
    
    #set up the neural network with default arguments
    network = PFN( (50, 9), ( 2, ) )

    #train with default arguments
    network.trainModel( sample, 'jetTagger.h5') 
