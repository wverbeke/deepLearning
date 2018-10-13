"""
functions for string manipulation that can be re-used 
"""

#remove the first character of a string if it is the given character
def removeLeadingCharacter( string , character):
    new_string = string
    if new_string and new_string[0] == character:
        new_string = new_string[1:]
    return new_string

