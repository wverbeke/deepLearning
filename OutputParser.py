import os
import operator
import sys
from stringTools import removeLeadingCharacter

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
    def __init__(self, output_directory):

        self.output_directory = output_directory
        self.AUC_map = {}

        #list all files in output directory
        sub_directories = listSubDirectories( self.output_directory )
        
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

    
    def outputName(self):
        output_name = self.output_directory 
        output_name = output_name.replace( 'output', '' )
        output_name = removeLeadingCharacter( output_name, '_' )
        return output_name

    
    def copyBestModelsOutput(self):
        best_model_directory = 'bestModels_{}'.format( self.outputName() )
        os.system('mkdir -p {}'.format( best_model_directory ) )
        for i, model in enumerate( self.ranked_models ):
            if i >= 10:
                break
            os.system('cp -r {0}/{1} {2}/model_rank_{3}'.format( self.output_directory, model[0], best_model_directory, i + 1 ) )


    def bestModels(self):
        self.rankModels()
        self.copyBestModelsOutput()
        self.printBestModels()
       


if __name__ == '__main__' :
    
    if len( sys.argv ) == 2 :
        ranker = OutputParser()
        ranker.bestModels()
    else :
        print( 'Error: incorrect number of arguments given to script.')
        print( 'Usage: <python OutputParser.py output_directory>')
     
