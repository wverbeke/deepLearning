import numpy as np

#import other parts of code 
from Trait import IntTraitClassFactory, FloatTraitClassFactory, StringTraitClassFactory, BoolTrait


#A genome is a collection of different kinds of Traits 
class Genome:

    def __init__( self, traits ):
        self._traits = list( traits )

   
    def __repr__( self ):
        return self.__str__() 


    def __str__( self ):

        #make comma separated list 
        string_representation = ( self.__class__.__name__ + '(' )
        for trait in self._traits:
            string_representation += str(trait) + ', '

        #remove last comma and space 
        string_representation = string_representation[:-2] 

        string_representation += ')'
        return string_representation

    
    #randomize the entire Genome 
    def randomize( self ):
        for trait in self._traits:
            trait.mutate() 


    #mutate a random Trait in the Genome
    def mutate( self ):

        #randomnly choose trait to mutate 
        mutation_index = np.random.randint( len( self._traits ) )
        
        #mutate trait
        self._traits[mutation_index].mutate()
    

    #reproduce with an other Genome, creating two new Genomes 
    def reproduce( self, other ):
        new_traits = [ traits[0].newTraits(traits[1]) for traits in zip(self._traits, other._traits) ]
        new_genome_1 = Genome( traits[0] for traits in new_traits )
        new_genome_2 = Genome( traits[1] for traits in new_traits )
        return new_genome_1, new_genome_2

   
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

    genome = Genome( [number_of_nodes, depth, optimizer] ) 
    genome.randomize()
    genome.mutate()
    genome_2 = Genome( [number_of_nodes, depth, optimizer] ) 
    genome_2.mutate()
    print( genome.reproduce( genome_2 ) )
    
