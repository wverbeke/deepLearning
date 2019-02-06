import numbers
import sys
import itertools


#include other parts of framework
sys.path.insert(0, '../')
sys.path.insert(0, '../geneticAlgorithm')
from Trait import IntTraitClassFactory, FloatTraitClassFactory, StringTraitClassFactory, BoolTrait
from Genome import Genome
from Generation import Generation
from Configuration import *
 


#class that reads the metadata specifying the test, training and validation data sets
class TrainingDataReader( Configuration ):
    _required_parameters = [ 'root_file_name', 'signal_tree_name', 'background_tree_name', 'list_of_branches', 'weight_branch', 'only_positive_weights', 'validation_fraction', 'test_fraction' ]

    def __init__( self, configuration_file ):
        self._parameters = {}
        for key in self._required_parameters:
            if hasattr( configuration_file, key ):
                self._parameters[key] = getattr( configuration_file, key )
        
        #check whether all necessary keys are present 
        self._allParametersPresent()


    #overload of abstract function that is not used in this case
    def _removeRedundancies( self ):
        pass



#class that handles setting up a genetic algorithm from the corresponding input file 
#it creates a dictionary of Trait classes, allowing the creation of random Genomes, which can be converted to parameter configurations, and vice-versa
class GeneticAlgorithmInputReader:

    def __init__( self, configuration_file ):
        self._trait_classes = {}
        for key, value in configuration_file.parameter_ranges.items():

            if isinstance(value, tuple) and len(value) == 2:

                if all( isinstance(x, bool) for x in value ):
                    self._trait_classes[key] = BoolTrait

                elif all( isinstance(x, numbers.Real) for x in value ):
                    self._trait_classes[key] = FloatTraitClassFactory( *value )

            elif isinstance(value, list):

                if all( isinstance( x, numbers.Integral ) for x in value ):
                    self._trait_classes[key] = IntTraitClassFactory( value )

                elif all( isinstance( x, str ) for x in value ):
                    self._trait_classes[key] = StringTraitClassFactory( value )

                else:
                    raise TypeError('Error in ') 

            else:
                raise TypeError('')


    def randomGenome( self ):
        return Genome( { key : traitClass.randomTrait() for key, traitClass in self._trait_classes.items() } )

    
    def randomGeneration( self, size ):
        genome_generator = ( self.randomGenome() for i in range(size) )
        return Generation( genome_generator )

    
    #convert dictionary containing configuration values to Genome 
    def dictionaryToGenome( self, value_dict):

        #check if the appropriate keys are present 
        if value_dict.keys() != self._trait_classes.keys():
            raise KeyError('configuration file parameters and keys of passed dictionary are different')

        trait_dict = { name : self._trait_classes[name](value) for name, value in value_dict.items() }
        return Genome(trait_dict)


    
def genomeToNeuralNetworkConfiguration( genome ):
    configuration_dict = { name : trait.value() for name, trait in genome }
    return DenseNeuralNetworkConfiguration( **configuration_dict )


def generationToNeuralNetworkConfigurations( generation ):
    return list( genomeToNeuralNetworkConfiguration( genome ) for genome in generation )


def neuralNetworkConfigurationAndInputToGenome( configuration, inputReader):
    return inputReader.dictionaryToGenome( {name : parameter for name, parameter in configuration} )


def neuralNetworkConfigurationsAndInputToGeneration( configurations, inputReader):
    return Generation( neuralNetworkConfigurationAndInputToGenome( config, inputReader ) for config in configurations )
    


#class handling the input data for a grid scan over given neural network configurations
#the class will yield the unique configurations made from the input file one by one
class GridScanInputReader:

    def __init__( self, configuration_file ):

        #make dictionary mapping each parameter name to all possible options 
        _value_lists = {}
        for name, value_list in configuration_file.parameter_values.items():
        	_value_lists[name] = value_list
        
        #make a generator yielding dictionaries mapping the parameter names to each possible combination of options
        dict_generator = ( dict( zip( _value_lists.keys(), values ) ) for values in itertools.product( *_value_lists.values() ) )
        
        #make list with unique parameter combinations 
        self._configurations = list( set( newConfigurationFromDict( **input_dict ) for input_dict in dict_generator ) )


    def __len__( self ):
        return len( self._configurations )


    #yield a new configuration
    def __iter__( self ):
        for configuration in self._configurations:
            yield configuration



if __name__ == '__main__' :

    #code to test GeneticAlgorithmInputReader
    genetic_input_file = __import__('input_geneticAlgorithm')
    genetic_input_reader = GeneticAlgorithmInputReader( genetic_input_file )
    genome = genetic_input_reader.randomGenome()
    configuration = genomeToNeuralNetworkConfiguration( genome )
    genome_2 = neuralNetworkConfigurationAndInputToGenome( configuration, genetic_input_reader )
    generation = genetic_input_reader.randomGeneration( 1000 )
    configurations = generationToNeuralNetworkConfigurations( generation )
    new_generation = neuralNetworkConfigurationsAndInputToGeneration( configurations, genetic_input_reader )

    #code to test GridScanInputReader
    grid_input_file = __import__('input') 
    grid_input_reader = GridScanInputReader( grid_input_file )

    #code to test TrainingDataReader
    training_input_reader = TrainingDataReader( genetic_input_file )
