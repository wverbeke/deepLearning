import xgboost as xgb 

def trainGradientBoostedForestClassificationModel( train_data, train_labels, validation_data, validation_labels, train_weights = None, validation_weights = None, model_name = 'model', number_of_trees = 100, learning_rate = 0.1,  max_depth = 2, min_child_weight = 1, subsample = 0.5, colsample_bytree = 1, gamma = 0, alpha = 0, number_of_threads = 1):
    
    #convert training and validation data to dmatrix
    training_matrix = xgb.DMatrix( train_data, weight = train_weights, label = train_labels, nthread = number_of_threads)
    validation_matrix = xgb.DMatrix( validation_data, weight = validation_weights, label = validation_labels, nthread = number_of_threads)
    
    model_parameters = {
    	'learning_rate' : learning_rate,
    	'max_depth' : max_depth,
    	'min_child_weight' : min_child_weight,
    	'subsample' : subsample,
    	'colsample_bytree' : colsample_bytree,
    	'gamma' : gamma,
    	'alpha' : alpha,
        'nthread' : number_of_threads
    }
    
    #train classifier 
    booster = xgb.train( model_parameters, training_matrix, number_of_trees )
    
    #save model 
    booster.save_model( model_name + '.bin' )
