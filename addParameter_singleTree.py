from ROOT import TFile, TTree

#import other parts of framework
import os, sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )
from parametrization.ParameterAddition import ParameterAdderSingleTree
from miscTools.stringTools import canConvertToFloat


if __name__ == '__main__':

    if len( sys.argv ) < 5:
        print('Incorrect number of command line argument given. Aborting.')
        print('Usage : <python addParameters_singleTree.py root_file_name tree_name parameter_names background_default_values >')
        print('Or alternatively: <python ParematerAddition_singleTree.py root_file_name tree_name parameter_name default_value parameter_name_2 default_value_2 ... >')
        sys.exit()

    else:    
        input_file_name = sys.argv[1]
        root_file = TFile( input_file_name, 'update' )
        tree_name = sys.argv[2]
        tree = root_file.Get( tree_name )
        parameter_names = []
        background_defaults = []
        for entry in sys.argv[3:]:
            if canConvertToFloat( entry ):
                background_defaults.append( float( entry ) )
            else :
                parameter_names.append( entry )

        parameter_adder = ParameterAdderSingleTree( tree, parameter_names, background_defaults )
        parameter_adder.parametrizeTree( tree )
