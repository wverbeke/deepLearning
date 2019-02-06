import numpy as np

#import other parts of code 
from Trait import IntTraitClassFactory, FloatTraitClassFactory, StringTraitClassFactory, BoolTrait


#A genome is a collection of different kinds of Traits 
class Genome:

    def __init__( self, trait_dict):
        self._trait_dict = dict( trait_dict )

   
    def __repr__( self ):
        return self.__str__() 


    def __str__( self ):

        #make comma separated list 
        string_representation = ( self.__class__.__name__ + '(' )
        for trait_name, trait in self._trait_dict.items():
            string_representation += trait_name + ' = ' + str(trait) + ', '

        #remove last comma and space 
        string_representation = string_representation[:-2] 

        string_representation += ')'
        return string_representation


    #iterate over dictionary in Genome 
    def __iter__( self ):
        for name, trait in self._trait_dict.items():
            yield name, trait
    

    #randomize the entire Genome 
    def randomize( self ):
        for trait in self._trait_dict.values():
            trait.mutate() 


    #mutate a random Trait in the Genome
    def mutate( self ):

        #randomnly choose trait to mutate 
        mutation_key = np.random.choice( list(self._trait_dict.keys()) )
        
        #mutate trait
        self._trait_dict[mutation_key].mutate()
    

    #reproduce with an other Genome, creating two new Genomes 
    def reproduce( self, other ):

        #check that self and other have the same keys in their Trait dictionary
        if self._trait_dict.keys() != other._trait_dict.keys():
            raise KeyError('Two genomes with different sets of Traits can not reproduce.') 

        #make two new dictionaries of Traits
        new_trait_dict_1 = {}
        new_trait_dict_2 = {}
        for key in self._trait_dict:
            new_trait_tuple = self._trait_dict[key].newTraits( other._trait_dict[key] ) 
            new_trait_dict_1[key] = new_trait_tuple[0]
            new_trait_dict_2[key] = new_trait_tuple[1]
        return Genome( new_trait_dict_1 ), Genome( new_trait_dict_2 )
            

    #compute the fitness of this Genome depending on some provided function to determine the Fitness  
    def fitness( self, fitness_func ):
        return fitness_func( self )
        


if __name__ == '__main__':

    #Code to test the Genome class 
    NumberOfNodes  = IntTraitClassFactory( range( 1024 ) )
    Depth = IntTraitClassFactory( range( 10 ) )
    Optimizer = StringTraitClassFactory( ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'] ) 

    number_of_nodes = NumberOfNodes(0)
    depth = Depth(0)
    optimizer = Optimizer('Nadam')

    genome = Genome( {'number_of_nodes' : number_of_nodes, 'depth' : depth, 'optimizer' : optimizer } ) 
    genome.randomize()
    genome.mutate()
    genome_2 = Genome( {'number_of_nodes' : number_of_nodes, 'depth' : depth, 'optimizer' : optimizer } ) 
    genome_2.mutate()
    print( genome.reproduce( genome_2 ) )
    
