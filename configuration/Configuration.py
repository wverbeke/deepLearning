import abc


#abstract base class representing the configuration of a machine learning algorithm
class Configuration( abc.ABC ):
    
    required_parameters = []
    def __init__( self, **input_configuration ):
        self.parameters = {}
        for key, value in input_configuration.items():
            #setattr( self, key, value )
            self.parameters[key] = value

        #check whether all necessary keys are present 
        self._allParametersPresent()

        #check whether any rogue 
        self._noRogueParametersPresent()

        #remove reduncancies in the 
        self._removeRedundancies() 
   
 
    @abc.abstractmethod 
    def _removeRedundancies( self ):
        '''Function that sets configurations that are equivalent to a default representation so the multiple trainings with the same configuration can be avoided.'''
   
 
    def _allParametersPresent( self ):
        '''Function that checks whether all necessary parameters for setting up the training are present.'''
        for key in self.required_parameters:
            if key not in self.parameters:
                raise KeyError('Required parameter {} is not present in input configuration'.format( key ) )

    
    def _noRogueParametersPresent( self ):
        '''Function that checks whether no unknown parameters are present that are not needed to set up the training.'''
        for key in self.parameters:
            if key not in self.required_parameters:
                raise KeyError('Parameter {} key should not be present in configuration of type {}'.format( key, self.__class__.__name__ ) )


    def getParameterSet(self):
    	return tuple( a.values() )
    
    
    def __hash__( self ):
    	return hash( self.getParameterSet() )
    	
    
    def __eq__( self, other):
    	return ( self.getParameterSet() == other.getParameterSet() )
    
    
    def __ne__( self, other ):
    	return not( self.__eq__( other ) )

    
    def __str__( self ):
        return str( self.parameters )



#configuration of a Dense neural network
class DenseNeuralNetworkConfiguration( Configuration ):
    required_parameters = ['num_hidden_layers', 'units_per_layer', 'optimizer', 'learning_rate', 'learning_rate_decay', 'dropout_first', 'dropout_all', 'dropout_rate']
    
    def _removeRedundancies( self ):
    	if self.parameters['dropout_all'] and self.parameters['dropout_first']:
    		self.parameters['dropout_first'] = False
    	
    	if not( self.parameters['dropout_all'] or self.parameters['dropout_first'] ):
    		self.parameters['dropout_rate'] = 0
		
