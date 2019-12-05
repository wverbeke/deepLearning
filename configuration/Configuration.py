import abc
import json
import numbers
import operator
from collections import OrderedDict

#import other parts of framework
import os, sys
main_directory = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
sys.path.insert( 0, main_directory )
from miscTools.stringTools import canConvertToFloat



#abstract base class representing the configuration of a machine learning algorithm
class Configuration( abc.ABC ):
    
    _required_parameters = set()
    def __init__( self, **input_configuration ):
        
        #parameter dictionary should be ordered to ensure consistent hashing of Configuration with the same parameters
        self._parameters = OrderedDict()
        for key, value in input_configuration.items():
            self._parameters[key] = value

        #check whether all necessary keys are present 
        self._allParametersPresent()

        #check whether any rogue parameters are present or not 
        self._noRogueParametersPresent()

        #remove reduncancies in the configuration
        self._removeRedundancies() 

        #make sure order is always the same for hashing, json package might not preserve this (Check!)
        self._parameters = OrderedDict( sorted( self._parameters.items(), key = operator.itemgetter(0) ) )

 
    def _removeRedundancies( self ):
        '''Function that sets configurations that are equivalent to a default representation so the multiple trainings with the same configuration can be avoided.'''
        pass 
   
 
    def _allParametersPresent( self ):
        '''Function that checks whether all necessary parameters for setting up the training are present.'''
        for key in self._required_parameters:
            if key not in self._parameters:
                raise KeyError('Required parameter {} is not present in input configuration'.format( key ) )

    
    def _noRogueParametersPresent( self ):
        '''Function that checks whether no unknown parameters are present that are not needed to set up the training.'''
        for key in self._parameters:
            if key not in self._required_parameters:
                raise KeyError('Parameter {} key should not be present in configuration of type {}'.format( key, self.__class__.__name__ ) )


    def getParameterTuple(self):
    	return tuple( self._parameters.values() )
    
    
    def __hash__( self ):
    	return hash( self.getParameterTuple() )
    	
    
    def __eq__( self, other):
    	return ( self.getParameterTuple() == other.getParameterTuple() )
    
    
    def __ne__( self, other ):
    	return not( self.__eq__( other ) )

    
    def __str__( self ):
        string_repr =  self.__class__.__name__ + '('
        for key, value in self:
            string_repr += key + " = " + str(value) + ', '
        string_repr = string_repr[:-2]
        string_repr += ')'
        return string_repr

    
    def __repr__( self ):
        return self.__str__()


    def name( self ):

        model_name = ''

        #build model name by combining the parameters of the configuration
        for parameter_name in self._parameters:

            #convert the parameter value to a string
            value_name = str( self._parameters[parameter_name] )

            #for boolean attributes just add the name if attribute is True ( necessary to avoid filenames longer than OS can handle ) and write nothing if it is False
            if value_name == 'False':
                continue

            #add the name of the paramter for numeric paramters or boolean parameters that are True
            #for parameters that are strings, just the parameter value is added to the model name, not the parameter name.
            if canConvertToFloat( value_name ) or value_name == 'True' : 

                #we want to return a name where the parameter names are separated by underscores
                #if a parameter has a name including underscores, we will remove them and capitalize the parts of the name instead 
                parts = parameter_name.split('_')
                for i, part in enumerate( parts ):
                    if i > 0:
                        part = part.capitalize()
                    model_name += part 

                #add and extra '=' for numeric parameters ( to be followed by the value )
                if canConvertToFloat( value_name ):
                    model_name += '='

            #add value name except for boolean parameters
            if value_name != 'True':

                #replace . with p for floating point parameter values
                model_name += value_name.replace( '.', 'p' )
        
            #separate parameters by underscore 
            model_name += '_'

        #remove trailing underscore
        model_name = model_name[:-1]

        return model_name 
    
    
    def __iter__( self ):
        for name, parameter in self._parameters.items():
            yield name, parameter


    #write out configuration to json file 
    def toJSON( self, output_file_path ):
        with open(output_file_path , 'w') as f:
            json.dump( self._parameters, f )


    #initialize configuration from json file
    @classmethod
    def fromJSON( cls, input_file_path ):
        parameter_dict = {}
        with open( input_file_path ) as f:
            parameter_dict = json.load( f )
        return cls( **parameter_dict )


    def __getitem__( self, key ):
        return self._parameters[key] 

    
    def keys( self ):
        return self._parameters.keys()
        

#list of all possible configuration classes, this will be used to determine from the input which learning algorithm should be used 
_configuration_classes = []

#decorator to add each configuration class to the list
#Use this decorator when defining new configuration classes!
def registerConfiguration(cls):
    _configuration_classes.append(cls)
    return cls


#find the configuration class that expects all the keys present in the input dictionary
def findConfigurationClass( **input_dictionary ):
    
    input_keys = { key for key in input_dictionary }
    for cls in _configuration_classes:
        if input_keys == cls._required_parameters:
            return cls
    
    #if no appropriate class is found, raise and error
    raise KeyError('No configuration class expecting the inputs {} is found'.format( input_keys ) )


#make a new configuration object from a given input dictionary
def newConfigurationFromDict( **input_dictionary ):
    config_class = findConfigurationClass( **input_dictionary )
    return config_class( **input_dictionary )


#make a new configuration object from a given JSON file 
def newConfigurationFromJSON( input_file_path ):
    parameter_dict ={}
    with open( input_file_path ) as f:
        parameter_dict = json.load( f )
    return newConfigurationFromDict( **parameter_dict )

    
		
