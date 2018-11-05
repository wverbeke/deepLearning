import itertools

class OptimizerInfo

class Optimizer():
    optimizer_info = [
        #optimizer name, keras optimizer class, default learning rate, default decay rate 
        ('RMSprop', keras.optimizers.RMSprop, 0.001, 0.0),
        ('Adagrad', keras.optimizers.Adagrad, 0.01, 0.0),
        ('Adadelta', keras.optimizers.Adadelta, 1.0, 0.0),
        ('Adam', keras.optimizers.Adam, 0.001, 0.0),
        ('Adamax', keras.optimizers.Adamax, 0.002, 0.0),
        ('Nadam', keras.optimizers.Nadam, 0.002, 0.004)
    ]

    def __init__(optimizer_name, relative_learning_rate, relative_learning_rate_decay):
        
        #find correct index 
        index = 9999
        for i, entry in enumerate(optimizer_info):
            if optimizer_name == entry[0]:
                index = i
        if index == 9999:
            print('Error in Optimizer::__init__ : optimizer {} is not known. Returning control.'.format( optimizer_name ) )
            return
        
        self.optimizer_name = optimizer_name 
        
        default_learning_rate = optimizer_info[index][2]
        self.learning_rate = relative_learning_rate*default_learning_rate 
            
        default_learing_rate_decay = optimizer_info[index][3] 
        keras_optimizer = optimizer_info[index][1]
        if optimizer_name == 'Nadam' : 
            if relative_learning_rate_decay == 0:
                relative_learning_rate_decay = 1
            self.optimizer = keras_optimizer( lr = learning_rate, schedule_decay = default_learning_rate*relative_learning_rate_decay)
        else :
            self.optimizer = keras_optimizer( lr = learning_rate, decay = relative_learning_rate_decay) 

    
    def getOptimizer(self):
        return self.optimizer

    def getName(self):
        name = self.optimizer_name 
        name += '_learningRate{}'.format( self.learning_rate )
		name += ( '_learningRateDecay{}'.format( self.learning_rate_decay ) if (learning_rate_decay != d

	 model_name = 'model_{0}hiddenLayers_{1}unitsPerLayer_{2}_learningRate{3}'.format(num_hidden_layers, units_per_layer, activation, learning_rate)
    model_name += ( '_learningRateDecay{}'.format(learning_rate_decay) if (learning_rate_decay > 0) else '' )
    model_name += ( '_dropoutFirst{}'.format(dropout_rate) if dropout_first else '' )
    model_name += ( '_dropoutAll{}'.format(dropout_rate) if dropout_all else '' )


    
    def __eq__(self, rhs):
        return ( self.optimizer == self.optimizer )


    def __ne__(self, rhs):
        return not( self.__eq__(rhs) )


    def __hash__(self):
        return hash( self.optimizer )



class HyperParameterSet():
    def __init__(self, num_hidden_layers, units_per_layer, optimizer_name, learning_rate, learning_rate_decay, dropout_first, dropout_all, dropout_rate):
        self.num_hidden_layers = num_hidden_layers
        self.units_per_layer = units_per_layer
        self.optimizer = Optimizer( optimizer_name, learning_rate, learning_rate_decay ), 
        self.dropout_first = dropout_first
        self.dropout_all = dropout_all
        self.dropout_rate = dropout_rate

        #remove redudancies in variation 
        if self.dropout_all :
            self.dropout_first = False

        if not( self.dropout_first or self.dropout_all ):
            self.dropout_rate = 0


    def getParameterSet(self):
        return ( self.num_hidden_layers, self.units_per_layer, self.optimizer.getOptimizer(), self.dropout_first, self.dropout_all, self.dropout_rate )


    def __eq__(self, rhs):
        if self.getParameterSet() == rhs.getParameterSet() :
            return True
        return False


    def __ne__(self, rhs):
        return not( self.__eq__(rhs) )


    def __hash__(self):
        return hash( self.getParameterSet() )



class ConfigurationParser():
    def __init__(self, configuration_file):

        #list all variations specified in the input file
        option_list = [configuration_file.num_hidden_layers]
        option_list.append( configuration_file.units_per_layer )
        option_list.append( configuration_file.learning_rates )
        option_list.append( configuration_file.learning_rate_decays )
        option_list.append( configuration_file.dropout_first )
        option_list.append( configuration_file.dropout_all )
        option_list.append( configuration_file.dropout_rate )

        #make list of all combinations
        parameter_combinations = list( itertools.product( *option_list ) )
        self.configuration_list = []
        for parameter_tuple in parameter_combinations:
            self.configuration_list.append( HyperParameterSet( *parameter_tuple ) )

        #remove redundant( equal ) combinations from list 
        self.configuration_list = list( set( self.configuration_list ) )


    def numberOfConfigurations(self):
        return len( self.configuration_list )


    def yieldVariation(self):
        index = 0
        while index < self.numberOfConfigurations() :
            yield self.configuration_list[index].getParameterSet()
            index += 1

