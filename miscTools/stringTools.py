"""
functions for string manipulation that can be re-used 
"""

import os


#remove the first character of a string if it is the given character
def removeLeadingCharacter( string , character):
    new_string = string
    if new_string and new_string[0] == character:
        new_string = new_string[1:]
    return new_string


#check if a string can be converted to Float
def canConvertToFloat( string ):
    try :
        float( string )
        return True
    except ValueError:
        return False


#remove directory and file extension from path
def removePathAndExtension( path_to_file ):
    return os.path.basename( os.path.splitext( path_to_file )[0] )
