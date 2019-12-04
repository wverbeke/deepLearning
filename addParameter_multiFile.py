from ROOT import TFile, TTree

#import other parts of framework
import os, sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )
from parametrization.ParameterAddition import ParameterAdder



if __name__ == '__main__':

    if len( sys.argv ) < 5:
        print('Incorrect number of command line argument given. Aborting.')
        print('Usage : <python addParameters_multiTree.py signal_file_name signal_tree_name background_file_name background_tree_name parameter_names>')
        sys.exit()

    else:    
        signal_file_name = sys.argv[1]
        signal_file = TFile( signal_file_name )
        signal_tree_name = sys.argv[2]
        signal_tree = signal_file.Get( signal_tree_name ) 

        background_file_name = sys.argv[3]
        background_file = TFile( background_file_name, 'update' )
        background_tree_name = sys.argv[4]
        background_tree = background_file.Get( background_tree_name )

        parameter_names = sys.argv[5:]

        parameter_adder = ParameterAdder( signal_tree, parameter_names )
        parameter_adder.parametrizeTree( background_tree )
