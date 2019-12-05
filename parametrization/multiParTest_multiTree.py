import numpy as np
from ROOT import TFile, TTree
import uproot
import array
import matplotlib.pyplot as plt

from ParameterAddition import *


def generateRandomSignal():
    reco_array = np.random.uniform( 0, 10, ( 100, 1 ) )
    param_array = np.random.randn( 100, 2 )
    total_array = np.concatenate( [reco_array, param_array], axis = 1 )
    return total_array


def generateRandomBackground():
    reco_array = np.random.uniform( 0, 10, ( 100, 1 ) )
    return reco_array


def produceRootFile( file_name ):
    bkg_array = generateRandomBackground()
    sig_array = generateRandomSignal()

    f = TFile( file_name , 'RECREATE' )
    t = TTree( 'sig_tree', 'sig_tree' )
    arrays_to_fill = [ array.array( 'f', [0] ) for i in range( 3 ) ]
    branch_reco = t.Branch( 'reco', arrays_to_fill[0], 'reco/F' )
    branch_p1 = t.Branch( 'par1', arrays_to_fill[1], 'par1/F' )
    branch_p2 = t.Branch( 'par2', arrays_to_fill[2], 'par2/F' )

    for entry in sig_array:
        for i in range( len( arrays_to_fill ) ):
            arrays_to_fill[i][0] = entry[i]
        t.Fill()
    t.Write()   
    
    t_bkg = TTree( 'bkg_tree', 'bkg_tree' )
    branch_reco = t_bkg.Branch( 'reco', arrays_to_fill[0], 'reco/F' )
    for entry in bkg_array:
        arrays_to_fill[0][0] = entry 
        t_bkg.Fill()
    t_bkg.Write()
    f.Close()


def parametrizeRootFile( file_name ):
    f = TFile( file_name, 'update')
    t = f.Get('sig_tree')
    pa = ParameterAdder( t, ['par1', 'par2' ] )
    t_bkg = f.Get('bkg_tree' )
    pa.parametrizeTree( t_bkg ) 


def plotParameters( file_name ):
    f = uproot.open( file_name )
    t_sig = f[ 'sig_tree' ] 
    t_bkg = f[ 'bkg_tree' ]
    
    sig_par_arrays = t_sig.arrays( ['par1', 'par2'] )
    bkg_par_arrays = t_bkg.arrays( ['par1', 'par2'] )

    plt.subplot( 1, 2, 1 )
    plt.scatter( sig_par_arrays[b'par1'], sig_par_arrays[b'par2'] )
    plt.subplot( 1, 2, 2 )
    plt.scatter( bkg_par_arrays[b'par1'], bkg_par_arrays[b'par2'] )
    plt.show()
 
    

if __name__ == '__main__':
    produceRootFile( 'test.root' )
    parametrizeRootFile( 'test.root' ) 
    plotParameters( 'test.root' ) 
