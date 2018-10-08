from treeToArray import *
from ROOT import TFile
from Dataset import Data


if __name__ == '__main__':
    testFile = TFile("ttW_trainingData_new.root")
    testFile.ls()
    signalTree = testFile.Get('signalTree')
    backgroundTree = testFile.Get('bkgTree')
    branchList = [ branch.GetName() for branch in signalTree.GetListOfBranches() ]
    etaBranches = [ name for name in branchList if 'Eta' in name ]
    branchList = [ name for name in branchList if not 'Charge' in name ]
    branchList = [ name for name in branchList if not 'E' in name ]
    branchList += etaBranches
    branchList.remove('_weight')

    classification_data = Data('ttW_trainingData_new.root', 'signalTree', 'bkgTree', branchList, '_weight', 0.4, 0.2)

    classification_data.trainDenseClassificationModel(num_hidden_layers = 5, units_per_layer = 256, activation = 'relu', learning_rate = 0.0001, dropoutFirst=False, dropoutAll=False, dropoutRate = 0.5, num_epochs = 1, num_threads = 4)

