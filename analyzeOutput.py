import sys

#import other parts of code 
from output.OutputParser import OutputParser
from output.outputAnalysis import plotAUCVSParameters

if __name__ == '__main__':

	if len( sys.argv ) == 2 :

		#make OutputParser for this directory 
		output_directory = sys.argv[1]
		parser = OutputParser( output_directory )
		parser.bestModels()

		#make analysis plots for this OutputParser 
		plotAUCVSParameters( parser )
		
	else :
		print( 'Error: incorrect number of arguments given to script.')
		print( 'Usage: <python {} output_directory>'.format( __file__ ) )

