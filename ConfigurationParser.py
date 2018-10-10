import itertools

class HyperParameterSet():
    def __init__(self, num_hidden_layers, units_per_layer, learning_rate, dropout_first, dropout_all, dropout_rate):
        self.num_hidden_layers = num_hidden_layers
        self.units_per_layer = units_per_layer
        self.learning_rate = learning_rate
        self.dropout_first = dropout_first
        self.dropout_all = dropout_all
        self.dropout_rate = dropout_rate

        #remove redudancies in variation 
        if self.dropout_all :
            self.dropout_first = False

        if not( self.dropout_first or self.dropout_all ):
            self.dropout_rate = 0


    def getParameterSet(self):
        return ( self.num_hidden_layers, self.units_per_layer, self.learning_rate, self.dropout_first, self.dropout_all, self.dropout_rate )


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

