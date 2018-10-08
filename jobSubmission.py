"""
Functions for submitting jobs to the T2 cluster
"""

import time

def initializeJobScript( file_name ):
    script = open( file_name, 'w' )
    script.write('export SCRAM_ARCH=slc6_amd64_gcc630\n')
    script.write('cd /user/${USER}/${CMSSW_BASE}/src\n')
    script.write('source /cvmfs/cms.cern.ch/cmsset_default.sh\n')
    script.write('eval `scram runtime -sh`\n')

    #inside the job switch back to directory where program was executed 
    current_directory = os.path.dirname( os.path.abspath( __file__ ) )
    script.write('cd ' + currentDir + '\n')

    #return the script
    return script


#submit script of given name as a job with given wall-time
def submitJobScript( script_name, wall_time = '24:00:00', num_threads = 1):

    #keep attempting submission until it succeeds
    while True:
        submission_command = 'qsub {0} -l walltime={1}'.format( script_name, wall_time )
        if num_threads > 1:
            submussion_command += ' -lnodes=1:ppn={}'.format(num_threads)
       
        #run submission command  
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
            print( output )
            break
        time.sleep(1)
