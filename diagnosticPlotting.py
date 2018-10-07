import matplotlib.pyplot as plt
import numpy as np
import root_numpy
from ROOT import TH1D, TCanvas, TLegend


def plotComparison( training_history, model_name, metric ):

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
    plt.savefig( metric + '_' + model_name + '.pdf')
    
    #clear canvas
    plt.clf()
    

def plotAccuracyComparison( training_history, model_name ):
    return plotComparison(training_history, model_name, 'acc')


def plotLossComparison( training_history, model_name ):
    return plotComparison(training_history, model_name, 'loss')


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


def plotROC(sig_eff, bkg_eff, model_name):
    
    #plot background rejection as a function of signal efficiency
    plt.plot( sig_eff, backgroundRejection(bkg_eff) , 'b')
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background rejection')
    plt.savefig('roc_' + model_name + '.pdf') 

    #clear canvas
    plt.clf()


#compute area under the ROC curve, a strong metric for evaluating the model performance
def areaUndeCurve(sig_eff, bkg_eff):
    
    #use trapezoidal rule to compute integral
    integral = np.trapz( backgroundRejection(bkg_eff), sig_eff)

    #integral could be negative depending on order of efficiency arrays (i.e. if it starts at high efficiency)
    return abs( integral )
    

def testOutputPlot( outputs_sig, weights_sig, outputs_bkg, weights_bkg):
    min_output = min( np.min( outputs_sig ), np.min( outputs_bkg ) )
    max_output = max( np.max( outputs_sig ), np.max( outputs_bkg ) )
    addHist( outputs_sig, weights_sig, 30, min_output, max_output, color = 'blue')
    addHist( outputs_bkg, weights_bkg, 30, min_output, max_output, color = 'red')
    bottom, top = plt.ylim()
    plt.ylim( 0,  top)
    plt.savefig('test_hist.pdf')
    plt.clf()


def binWidth(num_bins, min_bin, max_bin):
    return float( max_bin - min_bin) / num_bins


def addHist( data, weights, num_bins, min_bin, max_bin, color = 'blue'):
    bin_width = binWidth( num_bins, min_bin, max_bin )
    n, bins, _ = plt.hist(data, bins=num_bins, range=(min_bin, max_bin), weights = weights/np.sum(weights), histtype='step', lw=2, color=color)
    bin_errors = errors( data, weights, num_bins, min_bin, max_bin)
    bin_centers = 0.5*(bins[1:] + bins[:-1]) 
    plt.errorbar(bin_centers, n, yerr=bin_errors, fmt='none', ecolor=color)


def errors( data, weights, num_bins, min_bin, max_bin ):
    bin_errors = np.zeros( num_bins )
    bin_width = binWidth( num_bins, min_bin, max_bin )
    for entry in data:
        bin_number = int( (entry - min_bin) // bin_width )
        if bin_number == num_bins :
            bin_number = num_bins - 1
        weight_entry =  weights[bin_number]
        bin_errors[bin_number] += weight_entry*weight_entry
    bin_errors = (bin_errors ** 0.5)
    scale = np.sum(weights)
    bin_errors /= scale
    return bin_errors 


#def plotOutputShapeComparison( outputs_signal_training, weights_signal_training, outputs_background_training, outputs_signal_testing, outputs_background_testing, model_name, num_bins = 30):
#    
#    max_output = max( np.max( outputs_signal_training ), np.max( outputs_background_training ), np.max( outputs_signal_testing ), np.max( outputs_background_testing ) )
#    min_output = min( np.min( outputs_signal_training ), np.min( outputs_background_training ), np.min( outputs_signal_testing ), np.min( outputs_background_testing ) )
#
#    hist_signal_training = TH1D( "Signal ( training sample )", "Signal ( training sample ); Model output", num_bins, min_output, max_output )
#    root_numpy.array2hist( outputs_signal_training, hist_signal_training )
#    hist_signal_training.Scale( hist_signal_training.GetSumOfWeights() )
#    hist_background_training = TH1D( "Background ( training sample )", "Background ( training sample ); Model output", num_bins, min_output, max_output )
#    root_numpy.array2hist( outputs_background_training, hist_background_training )
#    hist_background_training.Scale( hist_background_training.GetSumOfWeights() )
#    hist_signal_testing = TH1D( "Signal ( test sample )", "Signal ( test sample ); Model output", num_bins, min_output, max_output )
#    root_numpy.array2hist( outputs_signal_testing, hist_signal_testing )
#    hist_signal_testing.Scale( hist_signal_testing.GetSumOfWeights() )
#    hist_background_testing = TH1D( "Background ( test sample )", "Background ( test sample ); Model output", num_bins, min_output, max_output )
#    root_numpy.array2hist( outputs_background_testing, hist_background_testing )
#    outputs_background_testing.Scale( outputs_background_testing.GetSumOfWeights() )
#
#    #make legend 
#    legend = TLegend( 0.25, 0.73, 0.87, 0.92, NULL, "brNDC"); 
#    legend.SetNColumns(2);
#    legend.SetFillStyle(0); 
#    legend.AddEntry( hist_signal_training, 'Signal ( training sample )' )
#    legend.AddEntry( hist_background_training, 'Background ( training sample )' )
#    legend.AddEntry( hist_signal_testing, 'Signal ( test sample )' )
#    legend.AddEntry( hist_background_testing, 'Background ( test sample )' )
#    
#    c = TCanvas("", "", 500, 500)
#    hist_signal_training.Draw('histe')
#    hist_background_training.Draw('histesame')
#    hist_signal_testing.Draw('histesame')
#    hist_background_testing.Draw('histesame')
#    legend.Draw('same')
#    
#    c.SaveAs( 'outputShape_' + model_name + '.pdf' )
