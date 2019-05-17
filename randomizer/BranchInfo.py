import numpy as np


#check if a branch corresponds to a jagged array, signifying that there are multiple values (objects) per event
def branchIsJagged( uproot_branch ):
    lazy_array = uproot_branch.lazyarray()
    return ( lazy_array.dtype == object and len( lazy_array.shape ) == 1 )



class BranchInfo:
    
    def __init__( self, uproot_branch):
        self.__name = uproot_branch.name.decode( 'utf-8' )
        self.__dtype = uproot_branch.lazyarray()[0].dtype
        self.__isJagged =  branchIsJagged( uproot_branch )

        if self.__isJagged :
            self.__dimensionality = uproot_branch.lazyarray()[0].shape[1:]
        else:
            self.__dimensionality = uproot_branch.lazyarray().shape[1:]

        self.__array = np.empty( self.__dimensionality, self.__dtype )

        #make sure array is at least 1D (i.e. array with 1 entry for scalar branches )
        if len(self.__dimensionality) == 0:
            self.__array = np.empty( 1, self.__dtype )
    

    root_type_map = {
    	'bool' : 'O',
    	'float64' : 'D',
    	'float32' : 'F',
    	'uint32' : 'i',
    	'uint64' : 'l',
    	'int32' : 'I',
    	'int64' : 'L'
    }

    
    def __rootTypeStr( self ):
        return self.root_type_map[ str(self.__dtype) ]
    
    
    def __rootShapeStr( self ):
        shape_str = ''
        for dim in self.__dimensionality:
        	shape_str += '[{}]'.format( dim )
        return shape_str


    @property
    def name( self ):
        return self.__name
    
    
    def addBranch( self, root_tree ):
        root_tree.Branch( self.__name, self.__array, '{}{}/{}'.format( self.__name, self.__rootShapeStr(), self.__rootTypeStr() ) )

    
    def fillArray( self, array, event_index, object_index = None):
        if object_index is None and self.__isJagged:
            raise ValueError( 'object_index argument is required to fill a jagged branch')

        if object_index is None:
            if len( self.__dimensionality ) == 0:
                self.__array[0] = array[event_index]
            else:
                self.__array[:] = array[event_index][:]
        else:
            if len( self.__dimensionality ) == 0:
                self.__array[0] = array[event_index][object_index]
            else:
                self.__array[:] = array[event_index][object_index][:]
    


class BranchInfoCollection :

    def __init__( self, uproot_tree, branch_list = None):
        if branch_list is None:
            self.__branches = [ BranchInfo( uproot_tree[key] ) for key in uproot_tree.keys() ]
        else:
            self.__branches = [ BranchInfo( uproot_tree[key] ) for key in branch_list ]
        

    def fillArrays( self, array_dict, event_index, object_index = None):
        
        #check that keys in array_dict correspond with the names of the branches
        if set( branch.name for branch in self.__branches ) != set( array_dict.keys() ):
            raise KeyError( 'Trying to fill arrays for BranchInfoCollection with different keys from the branches in the collection.' )
        
        #fill corresponding array for each branch  
        for branch in self.__branches:
            branch.fillArray( array_dict[ branch.name ], event_index, object_index )

    
    def addBranches( self, root_tree ):
        for branch in self.__branches:
            branch.addBranch( root_tree )
