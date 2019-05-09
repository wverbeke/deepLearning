"""
Script that watches the progress of the genetic algorithm.
When one generation is done training, the genetic algorithm will be run to submit the next generation for training
"""
import time
import sys

#import other parts of framework
from jobSubmission.submitJob import runningJobs
from runTraining import submitTrainingJobs


class GeneticAlgorithmWatcher :

    def __init__( self, configuration_file_name, job_id_list ):
        self.configuration_file_name = configuration_file_name
        self.job_id_set = set( job_id_list )

    
    def jobsAreRunning( self ):
        for job_id in runningJobs():
            if job_id in self.job_id_set:
                return True
        return False

    
    def submitNextGeneration( self ):
        submitTrainingJobs( self.configuration_file_name )



if __name__ == '__main__' :

    if len( sys.argv ) < 2:
        print( 'Error: incorrect number of arguments given to script.')
        print( 'Usage: <python geneticAlgorithmWatcher.py configuration.py>')
 
    else:
        configuration_file_name = sys.argv[1]
        job_id_generator = ( job_id for job_id in sys.argv[2:] )
        watcher = GeneticAlgorithmWatcher( configuration_file_name, job_id_generator )

        while True:
            if not watcher.jobsAreRunning():
                watcher.submitNextGeneration()
                break
            else:
                time.sleep( 200 )    
