import numpy as np
import itertools

#import other parts of code 
from Genome import Genome 
from Trait import IntTraitClassFactory, FloatTraitClassFactory, StringTraitClassFactory


#A Generation represents a population of Genomes 
#A new generation can be made by reproducing and keeping the fittest genomes, given a fitness function
class Generation:

    def __init__( self, population ):
        self._population = list( population )


    def __iter__( self ):
        for genome in self._population :
            yield genome


    def mutate( self, probability ):
        
        #select genomes to mutate 
        mutation_decisions = np.random.choice( np.arange(2), size = len( self._population ), p = [ ( 1 - probability ), probability ] )

        #mutate the chosen genomes 
        for choice, genome in zip( mutation_decisions , self._population ):
            if choice :
                genome.mutate() 
            

    def newGeneration( self, fitness_func ):
        
        #order population by decreasing fitness 
        self._population.sort( key = lambda genome : genome.fitness( fitness_func ), reverse = True )

        #keep 50% fittest genomes
        number_of_survivors = len( self._population ) // 2

        #because reproduction makes new genomes in pairs of two we need an even number of survivors so we have to fill in an even number of new genomes
        if ( number_of_survivors % 2 ) != 0 :
            number_of_survivors += 1 

        survivors = self._population[:number_of_survivors]

        #compute reproduction probabilities for surviving genomes 
        reproduction_probabilities = self._reproductionProbabilities( number_of_survivors )

        #pick the survivors that will reproduce 
        number_born = len(self._population) - number_of_survivors 
        reproduction_indices = np.random.choice( np.arange( number_of_survivors ), size = number_born, p = reproduction_probabilities ) 

        #make new genomes per two
        new_genome_generator = ( self._population[ reproduction_indices[i] ].reproduce( self._population[ reproduction_indices[i + 1] ] ) for i in range(0, number_born, 2) )
        new_genomes = list( itertools.chain( *new_genome_generator ) )

        #new generation is the combination of survivors and new genomes
        return Generation( survivors + new_genomes )
        

    # randomize the entire generation, 
    def randomize( self ):
        for genome in self._population:
            genome.randomize() 


    #probability for reproduction, depending on the ranking as a function of fitness 
    #these probabilities are only correct for a ranked population
    def _reproductionProbabilities( self, number_of_survivors  ):
        reproduction_probabilities = ( number_of_survivors - np.arange( number_of_survivors ) ) / np.sum( np.arange( 1, number_of_survivors + 1) )
        return reproduction_probabilities 



if __name__ == '__main__' : 

    #some code to test the Generation class
    def fitness_func( genome ):
        return genome._traits[0]._value

    NumberOfNodes  = IntTraitClassFactory( range( 1024 ) )
    Depth = IntTraitClassFactory( range( 10 ) )
    Optimizer = StringTraitClassFactory( ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'] )
    
    genomes = ( Genome( [NumberOfNodes(0), Depth(0), Optimizer('Nadam')] ) for i in range( 100 ) )
    
    generation = Generation( genomes )
    generation.randomize()

    for i in range(100):
        print( len( generation._population ) )
        generation = generation.newGeneration( fitness_func )
        generation.mutate( 0.2 )

    for genome in generation : 
        print( genome )


