import matplotlib.pyplot as plt

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
