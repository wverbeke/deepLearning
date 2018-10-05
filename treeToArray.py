import root_numpy
from ROOT import TFile 
import numpy as np

def treeToArray(tree, branchList):
    
    #convert tree to array of tuples 
    arrayTuples = root_numpy.tree2array(tree, branchList, '_weight>0')

    #convert to ndarray
    output_shape = ()
    num_rows = len(arrayTuples)
    #no column dimension if just one branch is specified 
    #num_cols = None if isinstance(branchList, basestring) else len(arrayTuples[0]) 

    argument_is_list = not isinstance(branchList, basestring ) 
    if argument_is_list :
        num_columns = len(arrayTuples[0])
        output_shape = (num_rows, num_columns)
    else :
        output_shape = (num_rows, )

    retArray = np.zeros( (output_shape) )
    for i, entry in enumerate(arrayTuples):
        if argument_is_list:
            entry = list(entry)
            retArray[i] = np.asarray( entry )
        else :
            retArray[i] = entry

    return retArray
    
def writeArrayToFile(array, fileName):
    np.save( fileName , array)

def loadArray(fileName):
    return np.load(fileName)

def listOfBranches(tree):
    names = [ branch.GetName() for branch in tree.GetListOfBranches() ]
    return names 

if __name__ == '__main__' :
    
    testFile = TFile("ttW_trainingData_new.root")
    testFile.ls()
    tree = testFile.Get('signalTree')
    names = [ branch.GetName() for branch in tree.GetListOfBranches() ]
    print( names )
    branchList = ['_jetPt1', '_jetEta1', '_jetPhi1', '_jetE1']
    array = treeToArray(tree, branchList)
    print( 'array.dtype = {}'.format( array.dtype ) )
    print('shape of array : {}'.format( array.shape ) )
    for entry in range(10):
        print('array[{0}] = {1}'.format( entry,  array[entry] ) )



