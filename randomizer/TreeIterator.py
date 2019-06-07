import uproot

#import other parts of framework
from eventSize import numberOfEventsToRead



class ArrayMap:

    def __init__( self, uproot_tree, entry_start, entry_stop, branch_list = None):
        self.__arrayDict = { key.decode('utf-8') : value for key, value in uproot_tree.arrays( branch_list, entrystart = entry_start, entrystop = entry_stop ).items() }
        self.__size = len( list( self.__arrayDict.values() )[0] )
    
    
    def __len__( self ):
        return self.__size
    
    
    @property
    def arrayDict( self ):
        return self.__arrayDict


    def keys( self ):
        return self.__arrayDict.keys()

    
    def values( self ):
        return self.__arrayDict.values()

        
    def __getitem__( self, key ):
        return self.__arrayDict[key]



class TreeIterator:

	def __init__( self, uproot_tree, branch_list = None ):
		self.__uproot_tree = uproot_tree
		self.__branch_list = branch_list 


	def __iter__( self ):
		number_of_events_to_read = numberOfEventsToRead( self.__uproot_tree )
		for i in range( 0, len( self.__uproot_tree ), number_of_events_to_read ):
			yield ArrayMap( self.__uproot_tree, i, i + number_of_events_to_read, self.__branch_list )
		
