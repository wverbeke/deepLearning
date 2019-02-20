"""
Tools to check the current CMSSW version, and make a new setup if no prior CMSSW setup is found, or it is outdated 
"""

import os

_recent_cmssw_release = 'CMSSW_10_2_11_patch1'



class CMSSWRelease:

    def __init__( self, version_name ):

        self.__version_name = version_name
        split_name = self.__version_name.split('_')

        if len(split_name) >= 4:
            self.__base_version = int( split_name[1] )
            self.__sub_version = int( split_name[2] )
            self.__sub_sub_version = int( split_name[3] )

        else:
            self.__base_version = 0
            self.__sub_version = 0
            self.__sub_sub_version = 0


    @property 
    def version_name( self ):
        return self.__version_name
    
    
    #comparison of different CMSSW versions
    def __lt__( self, other ):
        if self.__base_version < other.__base_version:
            return True
        if self.__sub_version < other.__sub_version:
            return True
        if self.__sub_sub_version < other.__sub_sub_version:
            return True
        return False 


    def __eq__( self, other ):
        return ( self.__base_version == other.__base_version and
            self.__sub_version == other.__sub_version and 
            self.__sub_sub_version == other.__sub_sub_version )


    def __gt__( self, other ):
        return not( self < other or self == other )


    def __le__( self, other ):
        return ( self < other or self == other )


    def __ge__( self, other ):
        return ( self > other or self == other )



def currentCMSSWRelease():
    try:
        return CMSSWRelease( os.environ['CMSSW_VERSION'] )
    except KeyError:
        return CMSSWRelease( '' )


def CMSSWVersionIsUpToDate():
    used_version = currentCMSSWRelease()
    target_version = CMSSWRelease( _recent_cmssw_release )
    return used_version >= target_version


def setupCMSSW( version_name ):
    os.system( 'cmsrel {}'.format( version_name ) )
    os.system( 'cd {}/src/; cmsenv'.format( version_name ) )


def getCMSSWDirectory():
    if CMSSWVersionIsUpToDate():
        pass
    else:
        setupCMSSW( _recent_cmssw_release )
    return os.environ['CMSSW_BASE']


if __name__ == '__main__':
    cmssw_1 = CMSSWRelease('CMSSW_10_4_12')
    cmssw_2 = CMSSWRelease('CMSSW_9_4_2')

    if cmssw_1 < cmssw_2:
        print('{} < {}'.format( cmssw_1.version_name, cmssw_2.version_name ) )

    if cmssw_1 >= cmssw_2:
        print('{} >= {}'.format( cmssw_1.version_name, cmssw_2.version_name ) )

    if cmssw_1 > cmssw_2:
        print('{} > {}'.format( cmssw_1.version_name, cmssw_2.version_name ) )

    if cmssw_1 == cmssw_2:
        print('{} == {}'.format( cmssw_1.version_name, cmssw_2.version_name ) )

    if cmssw_1 > cmssw_1:
        print('{} > {}'.format( cmssw_1.version_name, cmssw_1.version_name ) )

    if cmssw_1 >= cmssw_1:
        print('{} >= {}'.format( cmssw_1.version_name, cmssw_1.version_name ) )
