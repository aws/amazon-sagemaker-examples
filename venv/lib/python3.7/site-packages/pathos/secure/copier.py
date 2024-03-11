#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
#
# adapted from Mike McKerns' gsl SCPLauncher class
"""
This module contains the derived class for launching secure copy (scp)
commands.  See the following for an example.


Usage
=====

A typical call to a 'scp launcher' will roughly follow this example:

    >>> # instantiate the launcher, providing it with a unique identifier
    >>> copier = Copier('copier')
    >>>
    >>> # configure and launch the copy to the selected destination
    >>> copier(source='~/foo.txt', destination='remote.host.edu:~')
    >>> copier.launch()
    >>>
    >>> # configure and launch the copied file to a new destination
    >>> copier(source='remote.host.edu:~/foo.txt', destination='.')
    >>> copier.launch()
    >>> print(copier.response())
 
"""
__all__ = ['FileNotFound','Copier']

class FileNotFound(Exception):
    '''Exception for improper source or destination format'''
    pass

from pathos.connection import Pipe as _Pipe

# broke backward compatability: 30/05/14 ==> replace base-class almost entirely
class Copier(_Pipe):
    '''a popen-based copier for parallel and distributed computing.'''

    def __init__(self, name=None, **kwds):
        '''create a copier

Inputs:
    name: a unique identifier (string) for the launcher
    source: hostname:path of original  [user@host:path is also valid]
    destination: hostname:path for copy  [user@host:path is also valid]
    launcher: remote service mechanism (i.e. scp, cp)  [default = 'scp']
    options: remote service options (i.e. -v, -P)  [default = '']
    background: run in background  [default = False]
    decode: ensure response is 'ascii'  [default = True]
    stdin: file-like object to serve as standard input for the remote process
        '''
        self.launcher = kwds.pop('launcher', 'scp')
        self.options = kwds.pop('options', '')
        self.source = kwds.pop('source', '.')
        self.destination = kwds.pop('destination', '.')
        super(Copier, self).__init__(name, **kwds)
        return

    def config(self, **kwds):
        '''configure the copier using given keywords:

(Re)configure the copier for the following inputs:
    source: hostname:path of original  [user@host:path is also valid]
    destination: hostname:path for copy  [user@host:path is also valid]
    launcher: remote service mechanism (i.e. scp, cp)  [default = 'scp']
    options: remote service options (i.e. -v, -P)  [default = '']
    background: run in background  [default = False]
    decode: ensure response is 'ascii'  [default = True]
    stdin: file-like object to serve as standard input for the remote process
        '''
        if self.stdin is None:
            import sys
            self.stdin = sys.stdin
        for key, value in kwds.items():
            if key == 'command':
                raise KeyError('command')
            elif key == 'source': # if quoted, can be multiple sources
                self.source = value
            elif key == 'destination':
                self.destination = value
            elif key == 'launcher':
                self.launcher = value
            elif key == 'options':
                self.options = value
            elif key == 'background':
                self.background = value
            elif key == 'decode':
                self.codec = value
            elif key == 'stdin':
                self.stdin = value

        self._stdout = None
        self.message = '%s %s %s %s' % (self.launcher,
                                        self.options,
                                        self.source,
                                        self.destination)
        names=['source','destination','launcher','options','background','stdin','codec']
        return dict((i,getattr(self, i)) for i in names)

    # interface
    __call__ = config
    pass


if __name__ == '__main__':
    pass


# End of file
