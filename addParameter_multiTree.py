from ROOT import TFile, TTree

#import other parts of framework
import os, sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )
from parametrization.ParameterAddition import ParameterAdder



if __name__ == '__main__':

    if len( sys.argv ) < 5:
        print('Incorrect number of command line argument given. Aborting.')
        print('Usage : <python addParameters_multiTree.py root_file_name signal_tree_name background_tree_name parameter_names>')
        sys.exit()

    else:    
        input_file_name = sys.argv[1]
        root_file = TFile( input_file_name, 'update' )
        signal_tree_name = sys.argv[2]
        signal_tree = root_file.Get( signal_tree_name ) 
        background_tree_name = sys.argv[3]
        background_tree = root_file.Get( background_tree_name )
        parameter_names = sys.argv[4:]

        parameter_adder = ParameterAdder( signal_tree, parameter_names )
        parameter_adder.parametrizeTree( background_tree )
