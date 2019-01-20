import numpy as np
import abc 


#Abstract base class representing a single Trait of a genome to be used in genetic algorithms
class Trait( abc.ABC ): 

    def __init__( self, value ):
        self._value = value
        if not self._checkValue() :
            raise ValueError('Value provided in {} initialization is not among the possible values'.format( self.__class__.__name__ ) ) 


    def __str__( self ):
        return '{}({})'.format( self.__class__.__name__, self._value )
   
 
    def __repr__( self ):
        return self.__str__()

    
    @abc.abstractmethod
    def newTraits( self, other):
        """Make two new traits by mixing two traits."""


    @abc.abstractmethod 
    def mutate( self ):
        """Mutate the trait."""


    @abc.abstractmethod 
    def _checkValue( self ):
        """Returns boolean checking whether the current value is acceptable."""


    # return numeric representation of self._value
    # in most cases this will be the same as self._value, but it might be different when self._value is a string or other non-numeric object
    def _numericValue( self ):
        return self._value

    
    def value( self ):
        return self._value

    
    #This function should be used in the 'newTraits' method, it computes two new values based on previous values of two Traits 
    #depending on the subclass the returned values should be transformed to the correct type
    def _newParameters( self, other ):
        crossover_rate = np.random.uniform(0, 1)
        new_val_1 = (1 - crossover_rate)*self._numericValue() + crossover_rate*other._numericValue()
        new_val_2 = (1 - crossover_rate)*other._numericValue() + crossover_rate*self._numericValue() 
        return new_val_1, new_val_2

    

#Trait that is an integer number 
def IntTraitClassFactory( possibleValues ):

    class IntTrait( Trait ):
        _possibleValues = np.array( possibleValues, dtype = int)

        def newTraits( self, other ):
            new_val_1, new_val_2 = self._newParameters( other )
            return IntTrait( round(new_val_1) ), IntTrait( round(new_val_2) )


        def mutate( self ):
            self._value = np.random.choice( self._possibleValues )


        def _checkValue( self ):
            return ( self._value in self._possibleValues ) 


    return IntTrait



#boolean Trait, equivalent to integer Trait with 2 possible values
class BoolTrait( IntTraitClassFactory( range(2) ) ):
    pass 



#Trait that is an float between two values 
def FloatTraitClassFactory( min_val, max_val ):

    class FloatTrait( Trait ):
        _min = min_val
        _max = max_val

        def newTraits( self, other ):
            new_val_1, new_val_2 = self._newParemeters( other )
            return FloatTrait( new_val_1 ), FloatTrait( new_val_2 )


        def mutate(self):
            self._value = np.random.uniform( _min, _max )

        
        def _checkValue( self ):
            return ( self._value >= self._min and self._value <= self._max )


    return FloatTrait
        


#Trait that is a string 
def StringTraitClassFactory( possibleValues ):

    class StringTrait( Trait ):

        _possibleValues = np.array( possibleValues, dtype = object )
        nameToIndex = { name : i for i, name in enumerate( _possibleValues ) }
        indexToName = { i : name for i, name in enumerate( _possibleValues ) }

        def __init__( self, value ):

            #call base version of initializer
            super().__init__( value )

            #store integer index representing the string 
            self._index = self.nameToIndex[ self._value ]
            

        def _numericValue( self ):
            return self._index


        def newTraits( self, other ) :
            new_val_1, new_val_2 = self._newParameters( other )
            new_name_1, new_name_2 = self.indexToName[ round(new_val_1) ], self.indexToName[ round(new_val_2) ]
            return StringTrait( new_name_1 ), StringTrait( new_name_2 )


        def mutate( self ):
            self._value = np.random.choice( self._possibleValues )
            self._index = self.nameToIndex[ self._value ]


        def _checkValue( self ):
            return ( self._value in  self._possibleValues )


    return StringTrait



if __name__ == '__main__':

    #make testing code here
    pass 
