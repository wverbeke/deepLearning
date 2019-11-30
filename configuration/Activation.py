import keras 



class Activation():

    def __init__( self, activation_name ) :
        activationLayerDict = {
            'relu' : keras.layers.ReLU,
            'prelu' : keras.layers.PReLU,
            'leakyrelu' : keras.layers.LeakyReLU,
            'elu' : keras.layers.ELU
        }

        if not activation_name in activationLayerDict:
            raise KeyError( 'Error in Activation::__init__() : activation {} not found.'.format( activation_name ) )

        self.__keras_activation_layer = activationLayerDict[ activation_name ]


    def kerasActivationLayer( self ):
        return self.__keras_activation_layer
