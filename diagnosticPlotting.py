import matplotlib.pyplot as plt
import numpy as np

def plotComparison( training_history, network_name, metric ):

    #extract history of the metric 
    training_metric = training_history.history[metric]
    validation_metric = training_history.history['val_' + metric]

    #plot metric as a function of the training epoch
    epochs = range(1, len(training_metric) + 1 )
    plt.plot(epochs, training_metric, 'b', label = 'training ' + metric ) # this name can be improved 
    plt.plot(epochs, validation_metric, 'r', label = 'validation ' + metric ) #this name can be improved 
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(metric) #this name can be improved 
    plt.savefig( metric + '_' + network_name + '.pdf')
    
    #clear canvas
    plt.clf()
    

def plotAccuracyComparison( training_history, network_name ):
    return plotComparison(training_history, network_name, 'acc')

def plotLossComparison( training_history, network_name ):
    return plotComparison(training_history, network_name, 'loss')

def computeROC(outputs_signal, weights_signal, outputs_background, weights_background, num_points = 1000):
    
    sig_eff = np.zeros(num_points)
    bkg_eff = np.zeros(num_points)

    denominator_signal = np.sum( weights_signal )
    denominator_background = np.sum( weights_background )
    min_output = min( np.min( outputs_signal ), np.min(outputs_background ) )
    max_output = max( np.max( outputs_signal ), np.max(outputs_background ) )
    output_range = max_output - min_output

    for i in range(num_points):
        cut = min_output + (output_range/num_points)*i

        pass_signal = ( outputs_signal > cut ).reshape( len( weights_signal ) )
        numerator_signal = np.sum( weights_signal[ pass_signal ] )

        pass_background = ( outputs_background > cut ).reshape( len( weights_background ) )
        numerator_background = np.sum( weights_background[ pass_background ] )

        sig_eff[i] = numerator_signal/denominator_signal
        bkg_eff[i] = numerator_background/denominator_background

    return (sig_eff, bkg_eff)
    
def backgroundRejection( bkg_eff ):
    bkg_rejection =  np.ones( len(bkg_eff) ) - bkg_eff
    return bkg_rejection

def plotROC(sig_eff, bkg_eff, network_name):
    
    #plot background rejection as a function of signal efficiency
    plt.plot( sig_eff, backgroundRejection(bkg_eff) , 'b')
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background rejection')
    plt.savefig('roc_' + network_name + '.pdf') 

    #clear canvas
    plt.clf()


#compute area under the ROC curve, a strong metric for evaluating the model performance
def areaUndeCurve(sig_eff, bkg_eff):
    
    #use trapezoidal rule to compute integral
    integral = np.trapz( backgroundRejection(bkg_eff), sig_eff)

    #integral could be negative depending on order of efficiency arrays (i.e. if it starts at high efficiency)
    return abs( integral )
    

#def plotOutputShapeComparison():
