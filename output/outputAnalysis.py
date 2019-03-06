import numpy as np
import matplotlib.pyplot as plt

#add main directory
import os, sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )

from plotting.plotting import scatterPlot


def OutputParserToArrayDict( outputParser ):
    parameter_dict = {}
    for configuration in outputParser.configurations():
        for key, _ in configuration:
            if key in parameter_dict:
                parameter_dict[ key ].append( configuration[ key ] )
            else:
                parameter_dict[ key ] = [ configuration[ key ] ]

        auc = outputParser.getAUC( configuration )
        if 'auc' in parameter_dict:
            parameter_dict['auc'].append( auc )
        else :
            parameter_dict['auc'] = [ auc ]

    #convert all lists to numpy arrays 
    for key in parameter_dict:
        parameter_dict[ key ] = np.array( parameter_dict[key] )

    return parameter_dict


def plotAUCVSParameters( outputParser ):

    #output directory for plots 
    output_directory_name = 'scatterPlots_{}'.format( outputParser.analysisName() )
    
    #make output directory if it does not exist 
    if not os.path.isdir( output_directory_name ):
        os.makedirs( output_directory_name )

    #vectorize OutputParser and plot the auc as a function of each parameter
    parameter_dict = OutputParserToArrayDict( outputParser )
    for key in parameter_dict:

        #don't plot auc vs auc 
        if key == 'auc':
            continue

        #make name for x-label on plot 
        label_name = key.replace('_', ' ')
        label_name = label_name.capitalize()

        #make scatter plot 
        scatterPlot( parameter_dict[key], label_name, parameter_dict['auc'], 'AUC', os.path.join( output_directory_name, outputParser.analysisName() + '_AUC_VS_' + key ) )
