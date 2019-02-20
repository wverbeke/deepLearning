"""
Functions for submitting jobs to a cluster 
"""

import os
import time

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
    current_directory = os.path.dirname( os.path.abspath( __file__ ) )
    script.write('cd ' + current_directory + '\n')

    #return the script
    return script


def submitProcessJob( command_string, script_name, wall_time = '24:00:00', num_threads = 1):
    
    #make script
    script = newJobScript( script_name )

    #add command to script
    script.write( command_string )

    #close script
    script.close()

    #submit job
    #EXPAND THIS PART TO WORK FOR DIFFERENT JOB SUBMISSION SYSTEMS
    #IF POSSIBLE AUTOMATICALLY DETECT THE SUBMISSION SYSTEM
    submitQSubJob( script_name, wall_time, num_threads )



def testShellCommand( command_string ):

    #attempt to run command
    try: 
        subprocess.check_output(command_string , shell=True , stderr=subprocess.STDOUT)
        return True

    # command does not exist 
    except subprocess.CalledProcessError:
        return False



#submit script of given name as a job with given wall-time
def submitQsubJob( script_name, wall_time = '24:00:00', num_threads = 1):

    #keep attempting submission until it succeeds
    while True:
        submission_command = 'qsub {} -l walltime={}'.format( script_name, wall_time )
        if num_threads > 1:
            submission_command += ' -lnodes=1:ppn={}'.format(num_threads)
       
        #run submission command and pipe output to temporary file
        os.system( submission_command + ' > output_temp.txt 2>> output_temp.txt')

        #check output of submission command for errors 
        output = ''
        with open('output_temp.txt') as check:
            output = check.read()
        os.system( 'rm output_temp.txt' )
        
        error_messages = ['Invalid credential', 'Expired credential', 'Error']
        error_found = False
        for error in error_messages :
            if error in output:
                error_found = True
        if not error_found :
            first_line = output.split('\n')[0]
            print( first_line )
            break
        time.sleep(1)
