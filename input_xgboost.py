#name of input root file, relative to the directory of this script
root_file_name = 'ttW_trainingData_new.root'

#names of trees that contain signal and background events 
signal_tree_name = 'signalTree'
background_tree_name = 'bkgTree'

#list of variables to be used in training (corresponding to branches in the tree)
list_of_branches = [
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

#number of threads to use when training
number_of_threads = 1

#use genetic algorithm or grid-scan for optimization
use_genetic_algorithm = False

if use_genetic_algorithm:

    population_size = 500

    #ranges of neural network parameters for the genetic algorithm to scan
    parameter_ranges = {
        'number_of_trees' : list( range(100, 10000) ),
		'learning_rate' : (0.001, 1),
		'max_depth' : (2, 10),
		'min_child_weight' : (1, 20),
		'subsample' : (0.1, 1),
		'colsample_bytree' : (0.5, 1),
		'gamma' : (0, 1),
		'alpha' : (0, 1)
	}

else:
    parameter_values = {
        'number_of_trees' : [500, 1000, 2000, 4000, 8000],
		'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.5],
		'max_depth' : [2, 3, 4, 5, 6],
		'min_child_weight' : [1, 5, 10],
		'subsample' : [1],
		'colsample_bytree' : [0.5, 1],
		'gamma' : [0],
		'alpha' : [0]
	}
