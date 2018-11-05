import itertools

#import Optimizer class which collects the information on the optimizer that will be used 
from Optimizer import Optimizer 

class HyperParameterSet():
    def __init__(self, num_hidden_layers, units_per_layer, optimizer_name, relative_learning_rate, relative_learning_rate_decay, dropout_first, dropout_all, dropout_rate):
        self.num_hidden_layers = num_hidden_layers
        self.units_per_layer = units_per_layer
        self.optimizer = Optimizer( optimizer_name, relative_learning_rate, relative_learning_rate_decay ), 
        self.dropout_first = dropout_first
        self.dropout_all = dropout_all
        self.dropout_rate = dropout_rate

        #remove redudancies in variation 
        if self.dropout_all :
            self.dropout_first = False

        if not( self.dropout_first or self.dropout_all ):
            self.dropout_rate = 0

        #form name of this parameter combination
        self.name = 'model_{0}hiddenLayers_{1}unitsPerLayer_{2}'.format(num_hidden_layers, units_per_layer, 'relu')
        self.name += ( '_' + self.optimizer.name() )
        self.name += ( '_dropoutFirst{}'.format( dropout_rate ) if dropout_first else '' )
        self.name += ( '_dropoutAll{}'.format( dropout_all ) if dropout_all else '' )
        self.name = self.name.replace( '.', 'p' )


    def getParameterSet(self):
        return ( self.num_hidden_layers, self.units_per_layer, self.optimizer.optimizer(), self.dropout_first, self.dropout_all, self.dropout_rate )

	
    def getName(self):
    	return self.name 
    
    
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

