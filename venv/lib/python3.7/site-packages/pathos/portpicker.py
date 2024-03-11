#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
#
# this script prints out an available port number.
# adapted from J. Kim & M. McKerns utility functions
"""
This script prints out an available port number.
"""

class portnumber(object):
    '''port selector

Usage:
    >>> pick = portnumber(min=1024,max=65535)
    >>> print( pick() )
    '''

    def __init__(self, min=0, max=64*1024):
        '''select a port number from a given range.

The first call will return a random number from the available range,
and each subsequent call will return the next number in the range.

Args:
    min -- minimum port number  [default = 0]
    max -- maximum port number  [default = 65536]
        '''
        self.min = min
        self.max = max
        self.first = -1
        self.current = -1
        return

    def __call__(self):
        import random
        
        if self.current < 0: #first call
            self.current = random.randint(self.min, self.max)
            self.first = self.current
            return self.current
        else:
            self.current += 1
            
            if self.current > self.max:
                self.current = self.min
            if self.current == self.first: 
                raise RuntimeError( 'Range exhausted' )
            return self.current
        return

def randomport(min=1024, max=65536):
    '''select a random port number

Args:
    min -- minimum port number  [default = 1024]
    max -- maximum port number  [default = 65536]
    '''
    return portnumber(min, max)()


if __name__ == '__main__':

    pick = portnumber(min=1024,max=65535)
    print( pick() )


# End of file
