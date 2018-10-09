#name of input root file, relative to the directory of this script
root_file_name = 'ttW_trainingData_new.root'

#names of trees that contain signal and background events 
signal_tree = 'signalTree'
background_tree = 'bkgTree'

#list of variables to be used in training (corresponding to branches in the tree)
branch_list = [
    '_lepPt1', '_lepEta1', '_lepPhi1',
    '_lepPt2', '_lepEta2', '_lepPhi2',
    '_jetPt1', '_jetEta1', '_jetPhi1', '_jetCSV1',
    '_jetPt2', '_jetEta2', '_jetPhi2', '_jetCSV2',
    '_jetPt3', '_jetEta3', '_jetPhi3', '_jetCSV3',
    '_jetPt4', '_jetEta4', '_jetPhi4', '_jetCSV4',
    '_jetPt5', '_jetEta5', '_jetPhi5', '_jetCSV5',
    '_jetPt6', '_jetEta6', '_jetPhi6', '_jetCSV6',
    '_metPt1', '_metPhi1'
    ]

#branch that indicates the event weights 
weight_branch = '_weight'

#use only positive weights in training or not 
only_positive_weights = True

#validation and test fractions
validation_fraction = 0.4
test_fraction = 0.2

#different neural network parameters to test
num_hidden_layers = [2, 3, 4, 5, 6, 7, 8, 9, 10]
units_per_layer = [64, 128, 256, 512]
learning_rates = [0.001, 0.0001, 0.00001]
dropout_first = [False, True]
dropout_all = [False, True]
dropout_rate = [0.5, 0.3]

