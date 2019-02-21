import keras

class OptimizerInfo():

    def __init__(self, optimizer_name):
    	#optimizer name, keras optimizer class, default learning rate, default decay rate 
       	info = {
       		'RMSprop' : (keras.optimizers.RMSprop, 0.001, 0.0),
        	'Adagrad' : (keras.optimizers.Adagrad, 0.01, 0.0),
        	'Adadelta': (keras.optimizers.Adadelta, 1.0, 0.0),
        	'Adam' : (keras.optimizers.Adam, 0.001, 0.0),
        	'Adamax' : (keras.optimizers.Adamax, 0.002, 0.0),
        	'Nadam': (keras.optimizers.Nadam, 0.002, 0.004)
    	}

        if not (optimizer_name in info ):
            print( 'Error in OptimizerInfo::__init__() : optimizer {} not found. Returning control.'.format( optimizer_name ) )
        self.keras_optimizer = info[optimizer_name][0]
        self.default_learning_rate = info[optimizer_name][1]
        self.default_decay = info[optimizer_name][2]
    
    
    def defaultLearningRate(self):
        return self.default_learning_rate
    
    
    def defaultDecay(self):
    	return self.default_decay
    
    
    def kerasOptimizer(self):
    	return self.keras_optimizer
   
 

class Optimizer():

    def __init__(self, optimizer_name, relative_learning_rate = 1, relative_learning_rate_decay = 1):
        
        #make object of helper class to retrieve default information of optimizer
        optimizer_info = OptimizerInfo(optimizer_name)
        
        #set learning rate 
        self.learning_rate = relative_learning_rate*optimizer_info.defaultLearningRate()
        
        #set decay 
        default_decay = optimizer_info.defaultDecay()
        self.learning_rate_decay = relative_learning_rate_decay*default_decay
        
        #set keras optimizer 
        keras_optimizer = optimizer_info.kerasOptimizer()
        if optimizer_name == 'Nadam' :
            self.optimizer = keras_optimizer( lr = self.learning_rate, schedule_decay = self.learning_rate_decay )
        else :
        	self.optimizer = keras_optimizer( lr = self.learning_rate, decay = self.learning_rate_decay )
        
        #form optimizer name 
        #self.name = optimizer_name 
        #self.name += '_learningRate{}'.format( self.learning_rate )
        #self.name += ( '_learningRateDecay{}'.format( self.learning_rate_decay ) if (self.learning_rate_decay != default_decay) else '')


    def kerasOptimizer(self):
        return self.optimizer
    
    
    #def name(self):
    #	return self.name
    	
    
    def __eq__(self, rhs):
        return ( self.optimizer == self.optimizer )
    
    
    def __ne__(self, rhs):
        return not( self.__eq__(rhs) )
    
    
    def __hash__(self):
        return hash( self.optimizer )
