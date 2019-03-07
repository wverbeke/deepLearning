"""
Functions for submitting jobs to a cluster 
"""

import os
import time
import subprocess

#import other parts of framework
import sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) 
sys.path.insert( 0, main_directory )
from jobSubmission.CMSSW import getCMSSWDirectory


def newJobScript( script_name ):

    #set up CMSSW on worker node
    script = open( script_name, 'w' )
    script.write('cd {}/src\n'.format( getCMSSWDirectory() ) )
    script.write('source /cvmfs/cms.cern.ch/cmsset_default.sh\n')
    script.write('eval `scram runtime -sh`\n')

    #inside the job switch back to directory where program was executed 
    current_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
    script.write('cd ' + current_directory + '\n')

    #return the script
    return script


def submitProcessJob( command_string, script_name, wall_time = '24:00:00', num_threads = 1, high_memory = False):
    
    #make script
    script = newJobScript( script_name )

    #add command to script
    script.write( command_string )

    #close script
    script.close()

    #submit job
    #EXPAND THIS PART TO WORK FOR DIFFERENT JOB SUBMISSION SYSTEMS
    #IF POSSIBLE AUTOMATICALLY DETECT THE SUBMISSION SYSTEM
    return submitQsubJob( script_name, wall_time, num_threads, high_memory)


def runningJobs():
    #EXPAND THIS PART TO WORK FOR DIFFERENT JOB SUBMISSION SYSTEMS
    #IF POSSIBLE AUTOMATICALLY DETECT THE SUBMISSION SYSTEM
    return runningQsubJobs()


def testShellCommand( command_string ):

    #attempt to run command
    try: 
        subprocess.check_output( command_string , shell=True , stderr=subprocess.STDOUT )
        return True

    # command does not exist 
    except subprocess.CalledProcessError:
        return False



#submit script of given name as a job with given wall-time
def submitQsubJob( script_name, wall_time = '24:00:00', num_threads = 1, high_memory = False):

    #keep attempting submission until it succeeds
    submission_command = 'qsub {} -l walltime={}'.format( script_name, wall_time )
    
    if num_threads > 1:
        submission_command += ' -lnodes=1:ppn={}'.format(num_threads) 
    
    if high_memory :
        submission_command += ' -q highmem'
    
    while True:
        try:
            qsub_output = subprocess.check_output( submission_command, shell=True, stderr=subprocess.STDOUT )
        
        #submission failed, try again after one second 
        except subprocess.CalledProcessError as error:
            print('Caught error : "{}".\t Attempting resubmission.'.format( error.output.split('\n')[0] ) )
            time.sleep( 1 )
        
        #submission succeeded 
        else:
            first_line = qsub_output.split('\n')[0]
            print( first_line )
        
            #break loop by returning job id when submission was successful 
            return first_line.split('.')[0]
			

#check running qsub jobs 
def runningQsubJobs():

    #keep catching qsub errors untill command succeeds 
    while True:
        try:
            qstat_output = subprocess.check_output( 'qstat -u$USER', shell=True, stderr=subprocess.STDOUT )

        #qstat failed, try again after one second 
        except subprocess.CalledProcessError:
            time.sleep( 1 )

        else:
            return ( output_line.split('.')[0] for output_line in qstat_output.split('\n') )
