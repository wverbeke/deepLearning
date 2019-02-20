"""
functiones to list the content of a directory
"""
import os


def listContent( directory_name, typeCheck):
    for entry in os.listdir( directory_name ):
        full_path = os.path.join( directory_name, entry )
        if typeCheck( full_path ) :
            yield full_path


def listSubDirectories( directory_name ):
    return listContent( directory_name, os.path.isdir )


def listFiles( directory_name ):
    return listContent( directory_name, os.path.isfile )
