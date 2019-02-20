import os
import sys
import numpy as np
import time


#include other parts of framework
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )
from configuration.Configuration import newConfigurationFromJSON
from configuration.InputReader import *
from configuration.LearningAlgorithms import *
from output.OutputParser import OutputParser




if __name__ == '__main__' :
    
    begin_time = time.perf_counter() 

    #read input file 
    input_file = __import__('input_geneticAlgorithm')
    input_reader = GeneticAlgorithmInputReader( input_file )

    #make random generation
    generation_size = 10
    generation = input_reader.randomGeneration( generation_size )

    for genome in generation:
        print( genome )

    for i in range(10):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        configurations = generationToConfigurations( generation )

        #make output directory for testing
        output_directory_name = 'output_test'
        os.system( 'rm -r {}'.format( output_directory_name ) )
        os.system( 'mkdir -p {}'.format(output_directory_name) )

        print( 'number of configurations = {}'.format( len( configurations ) ) )
        write_counter = 0
        for configuration in configurations:
            subdir_name = 'output_{}'.format( configuration.name() )
            os.system( 'mkdir -p {}/{}'.format( output_directory_name, subdir_name ) )

            #write out json file with configuration
            configuration.toJSON( '{}/{}/{}.json'.format( output_directory_name, subdir_name, 'configuration_' + configuration.name() ) )

            #generate output file with random AUC value 
            with open( '{}/{}/{}'.format( output_directory_name, subdir_name, 'trainingOutput_{}.txt'.format( configuration.name() )  ), 'w' ) as trainingOutput:
                #trainingOutput.write('validation set ROC integral (AUC) = {}'.format( np.random.uniform( 0.5, 1 ) )  )
                trainingOutput.write('validation set ROC integral (AUC) = {}'.format( 0.5 + 0.5*configuration['units_per_layer']/1024 )  )

            write_counter += 1 

            new_configuration = newConfigurationFromJSON( '{}/{}/{}'.format( output_directory_name, subdir_name, 'configuration_' + configuration.name() + '.json' ) )
            
        print( 'write_counter = {}'.format(write_counter) )
            
        output_parser = OutputParser( output_directory_name )
        print( output_parser.analysisName() )
        generation = output_parser.toGeneration( input_reader )

        def fitness_func( genome ):
            config = genomeToConfiguration( genome )
            return output_parser.getAUC( config )

        print( 'old_generation_size = {}'.format(len(generation) ) )
        generation = generation.newGeneration( fitness_func )
        generation.mutate(0.3)
        print( 'new_generation_size = {}'.format( len( generation ) ) )

    #print final generation
    for genome in generation:
        print( genome )
    
    #print time it took to run 
    end_time = time.perf_counter() 
    print( 'Elapsed time = {}'.format( end_time - begin_time ) )
