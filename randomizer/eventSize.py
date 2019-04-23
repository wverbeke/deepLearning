"""
Functios to determine the size of an event and how many events should be read at once for large files
"""

#estimate size of single event in bytes 
def eventSize( uproot_tree ):
    branches = ( uproot_tree[key] for key in uproot_tree.keys() )
    total_size = 0
    for branch in branches:
        total_size += branch.uncompressedbytes()
    event_size = total_size / len( uproot_tree )
    return event_size


#number of events taking 1 GB of memory 
def numberOfEventsPerGB( event_size ):
    num_events = int( 1e9//event_size )
    return num_events


#number of events to read from file in one pass
def numberOfEventsToRead( uproot_tree, maximum_amount = 50000, maximum_size = 1):
    num_events_per_GB = numberOfEventsPerGB( eventSize( uproot_tree ) )
    return min( maximum_size*num_events_per_GB, maximum_amount )


#number of splittings in randomization depending on event size 
def numberOfFileSplittings( uproot_tree, maximum = 50000 ):
    number_of_events = len( uproot_tree )
    number_of_events_to_read = numberOfEventsToRead( uproot_tree, maximum )
    return max(1, round( number_of_events / number_of_events_to_read ) )
