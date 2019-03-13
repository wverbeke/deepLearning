import numpy as np
import matplotlib.pyplot as plt

def plotRange( array ):
    lower_bound = np.min( array )
    upper_bound = np.max( array )

    print( 'max = {}'.format( upper_bound ) )

    variation =( upper_bound - lower_bound )
    excess_space = variation*0.1

    lower_bound = (lower_bound - excess_space ) if lower_bound > 0 else ( lower_bound + excess_space )
    upper_bound = (upper_bound + excess_space ) if upper_bound > 0 else ( upper_bound - excess_space )

    return lower_bound, upper_bound
        

def scatterPlot( x_array, x_label, y_array, y_label, plot_file_name ):
    plt.xlabel( x_label )
    plt.ylabel( y_label )
    print( '~~~~~~~~~~~~~~~~~~~~~' )
    print( plot_file_name )
    print( plotRange( y_array ) )
    print( plotRange( x_array ) )
    plt.xlim( plotRange( x_array ) )
    plt.ylim( plotRange( y_array ) )
    plt.scatter( x_array, y_array )
    if not '.' in plot_file_name:
        plot_file_name += '.pdf'
    plt.savefig( plot_file_name )
    plt.clf()
