import os
import operator

def listContent( directory, typeCheck):
    content = []
    for entry in os.listdir( directory ):
        full_path = os.path.join( directory, entry )
        if typeCheck( full_path ) :
            content.append( full_path )
    return content


def listSubDirectories( directory ):
    return listContent( directory, os.path.isdir )


def listFiles( directory ):
    return listContent( directory, os.path.isfile )


class OutputParser:
    def __init__(self):

        self.AUC_map = {}

        #list all files in output directory
        sub_directories = listSubDirectories( 'output' )
        
        for directory in sub_directories : 
            files = listFiles( directory )
            files = [ f for f in files if f.split('.')[-1] == 'txt' ]
            
            for file_name in files :
                with open( file_name ) as f:
                    for line in f.readlines():
                        if 'AUC' in line :
                            line = line.replace('validation set ROC integral (AUC) = ', '')
                            AUC = float( line )
                            model_name = directory.split('/')[-1]
                            self.AUC_map[model_name] = AUC
                  
 
    def rankModels(self):
        self.ranked_models = sorted( list(self.AUC_map.items()),  key=operator.itemgetter(1), reverse=True )


    def printBestModels(self):
        for i, model in enumerate( self.ranked_models ):
            if i >= 10:
                break
            print( '########################################################') 
            print( 'Rank {}:'.format( i + 1 ) ) 
            print( model[0] )
            print( 'validation set ROC integral (AUC) = {}'.format( model[1] ) )


    def copyBestModelsOutput(self):
        os.system('mkdir -p bestModels')
        for i, model in enumerate( self.ranked_models ):
            if i >= 10:
                break
            os.system('cp -r output/{} bestModels/model_rank_{}'.format( model[0], i + 1 ) )

    def bestModels(self):
        self.rankModels()
        self.copyBestModelsOutput()
        self.printBestModels()
       


if __name__ == '__main__' :
    ranker = OutputParser()
    ranker.bestModels()
     
