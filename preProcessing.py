
#center entries of a 2D array around 0, with standard deviation 1 
def normalize2DArray( array ):

    #compute mean and standard deviation for each feature
    mean_array = np,mean( array, axis = 0 )
    std_array = np.std( array, axis = 0 )

    #apply transformations
    array = (array - std_array)/mean_array

    #return new array and the applied transformations 
    return ( array, mean_array, std_array )

