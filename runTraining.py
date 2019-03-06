import os
import sys
import subprocess


#import other parts of code 
from dataset.ModelTrainingSetup import ModelTrainingSetup

#from configuration import InputReader
from configuration.Configuration import newConfigurationFromDict
from configuration.LearningAlgorithms import *
from configuration.InputReader import *
from jobSubmission.submitJob import submitProcessJob 
from miscTools.listContent import listSubDirectories
from output.OutputParser import OutputParser


def trainAndEvaluateModel( training_configuration, model_configuration ):

    #make sure correct path is given for input root file
    root_file_name_full = os.path.join( os.path.dirname(os.path.abspath( __file__) ) , configuration_file.root_file_name )

    #train model
    classification_setup = ModelTrainingSetup( training_configuration )

    classification_setup.trainAndEvaluateModel( model_configuration )



def submitTrainingJob( configuration, number_of_threads, high_memory, output_directory):
    
    #model name 
    model_name = configuration.name() 
    
    #new directory for job output
    os.system('mkdir -p {}/{}'.format( output_directory, model_name ) )
    
    #make the command to run, starting with switching to the output directory
    command_string = 'cd {}/{}\n'.format( output_directory, model_name )
    
    #add the training to the command and make sure to refer to the correct directory for the runTraining file
    configuration_file_name = sys.argv[1]
    main_directory = os.path.dirname( os.path.abspath( __file__ ) )
    command_string += 'python {}/runTraining.py {}'.format( main_directory, configuration_file_name ) 
    for name, value in configuration:
        command_string += ' {}={}'.format( name, value )
    
    #pipe output to text files 
    log_file = 'trainingOutput_' + model_name + '.txt'
    error_file = model_name + '_err.txt'
    command_string += ' > {} 2>{} '.format( log_file, error_file)

    #dump configuration to output directory 
    configuration.toJSON( os.path.join( output_directory, model_name, 'configuration_' + model_name + '.json' ) )
    
    #submit this process 
    return submitProcessJob( command_string, 'trainModel.sh', wall_time = '24:00:00', num_threads = number_of_threads, high_memory = high_memory )


#convert string to either float, integer or boolean, or keep it as a string
def stringToArgumentType( string ):
    ret = string
    try:
        ret = int( ret )
    except ValueError:
    	try:
    		ret = float( ret )
    	except ValueError: 

            if ret == 'True':
                return True
            elif ret == 'False':
                return False
    return ret


#check how many generations of model trainings already happened when using a genetic algorithm
def lastGenerationNumber( output_directory_name ):
    subdirectories = ( subdir for subdir in listSubDirectories( output_directory_name ) if 'generation_' in subdir )
    generation_numbers = ( int( subdir.split('_')[-1] ) for subdir in subdirectories )
    
    #previous generations have been trained 
    try:
        return max( generation_numbers )
    
    #training the first generation
    except FileNotFoundError:
        os.system('mkdir {}'.format( output_directory_name ) )
        return 0
    except ValueError:
        return 0


def submitTrainingJobs( configuration_file_name ):

    configuration_file = __import__( configuration_file_name.replace('.py', '') )
    
    #list of neural network configurations to process 
    configuration_list = []
    
    #directory for job output
    output_directory_name = 'output_{}'.format( configuration_file_name.replace('input_', '').replace('.py', '') )
    
    #limit the number of model trainings that can be submitted 
    max_number_of_trainings = 2500
    number_of_models = 0
    
    #determine whether to run a genetic algorithm or a grid scan 
    if isGeneticAlgorithmInput( configuration_file ) :
    
        last_generation_number =  lastGenerationNumber( output_directory_name )
        new_generation_subdirectory = 'generation_{}'.format( last_generation_number + 1 )
    
        #set up genetic algorithm
        genetic_algo_configuration = GeneticAlgorithmInputReader( configuration_file )
        number_of_models = genetic_algo_configuration.population_size()
    
        #no previous generation has been trained yet, make a random population as the first generation
        if last_generation_number == 0:
            first_generation = genetic_algo_configuration.randomGeneration()
            configuration_list = generationToConfigurations( first_generation )
    
        #apply genetic algorithm to make the next generation
        else:
    
            #read output of previous generation 
            last_generation_subdirectory = 'generation_{}'.format( last_generation_number )
            output_last_generation = output_directory_name + '/' + last_generation_subdirectory
            output_parser = OutputParser( output_last_generation )
    
            #convert the output to a generation 
            #the fitness of a model is its AUC
            def fitness_func( genome ):
                config = genomeToConfiguration( genome )
                return output_parser.getAUC( config )
    
            generation = output_parser.toGeneration( genetic_algo_configuration )

            #evolve and mutate the generation
            generation = generation.newGeneration( fitness_func, target_size = configuration_file.population_size )
            generation.mutate( 0.2, 2 )
    
            configuration_list = generationToConfigurations( generation )
    
        #output directory name for next generation
        output_directory_name = output_directory_name + '/' + new_generation_subdirectory

    #grid scan
    else:
        grid_scan_configuration = GridScanInputReader( configuration_file )
        configuration_list = [ config for config in grid_scan_configuration ]
    
        number_of_models = len( grid_scan_configuration )
    
    #MAKE THIS MORE ROBUST BY CHECKING THE NUMBER OF RUNNING JOBS FOR THE USER
    if number_of_models > max_number_of_trainings :
        print( 'Error : requesting to train {} models. The cluster only allows {} jobs to be submitted.'.format( number_of_models, max_number_of_trainings ) )
        print( 'Please modify the configuration file to train less models.')
        sys.exit()

    #check if any job submission requirements were specified 
    number_of_threads = 1 
    if hasattr( configuration_file, 'number_of_threads' ):
        number_of_threads = configuration_file.number_of_threads
    high_memory = False
    if hasattr( configuration_file, 'high_memory' ):
        high_memory = configuration_file.high_memory
 
    #submit jobs and store their id
    job_id_list = []
    for configuration in configuration_list:
        job_id = submitTrainingJob( configuration, number_of_threads, high_memory, output_directory_name )
        job_id_list.append( job_id )
    
    if isGeneticAlgorithmInput( configuration_file ):
        watcher_command = 'nohup python geneticAlgorithmWatcher.py '
        watcher_command += configuration_file_name 
        for job in job_id_list:
            watcher_command += ' {}'.format( job ) 

        #piper output of watcher script to log file for debugging 
        watcher_output_file = 'watcher_generation{}_log.txt'.format( last_generation_number ) 
        
        #run the watcher script in the background
        subprocess.Popen( [ watcher_command + ' > {0} 2>> {0} &'.format( watcher_output_file ) ], shell = True )

    print( '########################################################' )
    print( 'Submitted {} neural networks for training.'.format( number_of_models ) )
    print( '########################################################' )

      


if __name__ == '__main__' :

    #check if atleast one additional argument is given, the program expects a configuration file to run
    if len( sys.argv ) < 2:
        print( 'Error: incorrect number of arguments given to script.')
        print( 'Usage: <python runTraining.py configuration.py>')
        sys.exit()

    #read configuration file 
    configuration_file_name = sys.argv[1]
    configuration_file = __import__( configuration_file_name.replace('.py', '') )

    if len( sys.argv ) > 2:    

        #build dictionary from command line arguments
        configuration_dict = {}
        for argument in sys.argv[2:] :
            if not '=' in argument:
                raise ValueError('Each command line must be of the form x=y in order to form a dictionary')
        
            key, value = argument.split('=')
            configuration_dict[key] = stringToArgumentType( value )
        
        #build model configuration from dictionary
        configuration = newConfigurationFromDict( **configuration_dict )
        
        #train model from configuration 
        trainAndEvaluateModel( TrainingDataReader( configuration_file), configuration )

    else :
        submitTrainingJobs( configuration_file_name )
