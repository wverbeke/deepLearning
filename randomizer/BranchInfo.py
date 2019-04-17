import numpy as np

class BranchInfo:
    
    def __init__( self, uproot_branch ):
        self.__name = uproot_branch.name.decode( 'utf-8' )
        self.__dtype = uproot_branch.lazyarray().dtype
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

    
    def fillArray( self, array, index ):
        if len( self.__dimensionality ) == 0:
            self.__array[0] = array[index]
        else:
            self.__array[:] = array[index][:]
    


class BranchInfoCollection :

    def __init__( self, uproot_tree ):
        self.__branches = [ BranchInfo( uproot_tree[key] ) for key in uproot_tree.keys() ]
        

    def fillArrays( self, array_dict, index ):
        
        #check that keys in array_dict correspond with the names of the branches
        if set( branch.name for branch in self.__branches ) != set( array_dict.keys() ):
            raise KeyError( 'Trying to fill arrays for BranchInfoCollection with different keys from the branches in the collection.' )
        
        #fill corresponding array for each branch  
        for branch in self.__branches:
            branch.fillArray( array_dict[ branch.name ], index )

    
    def addBranches( self, root_tree ):
        for branch in self.__branches:
            branch.addBranch( root_tree )
