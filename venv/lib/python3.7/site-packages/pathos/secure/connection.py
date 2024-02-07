#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
#
# adapted from Mike McKerns' and June Kim's gsl SSHLauncher class
"""
This module contains the derived class for secure shell (ssh) launchers
See the following for an example.


Usage
=====

A typical call to a 'ssh pipe' will roughly follow this example:

    >>> # instantiate the pipe, providing it with a unique identifier
    >>> pipe = Pipe('launcher')
    >>>
    >>> # configure the pipe to perform the command on the selected host
    >>> pipe(command='hostname', host='remote.host.edu')
    >>>
    >>> # execute the launch and retrieve the response
    >>> pipe.launch()
    >>> print(pipe.response())
 
"""
__all__ = ['Pipe']

from pathos.connection import Pipe as _Pipe

# broke backward compatability: 30/05/14 ==> replace base-class almost entirely
class Pipe(_Pipe):
    '''a popen-based ssh-pipe for parallel and distributed computing.'''

    def __init__(self, name=None, **kwds):
        '''create a ssh pipe

Inputs:
    name: a unique identifier (string) for the pipe
    host: hostname to recieve command [user@host is also valid]
    command: a command to send  [default = 'echo <name>']
    launcher: remote service mechanism (i.e. ssh, rsh)  [default = 'ssh']
    options: remote service options (i.e. -v, -N, -L)  [default = '']
    background: run in background  [default = False]
    decode: ensure response is 'ascii'  [default = True]
    stdin: file-like object to serve as standard input for the remote process
        '''
        self.launcher = kwds.pop('launcher', 'ssh')
        self.options = kwds.pop('options', '')
        self.host = kwds.pop('host', 'localhost')
        super(Pipe, self).__init__(name, **kwds)
        return

    def config(self, **kwds):
        '''configure a remote command using given keywords:

(Re)configure the copier for the following inputs:
    host: hostname to recieve command [user@host is also valid]
    command: a command to send  [default = 'echo <name>']
    launcher: remote service mechanism (i.e. ssh, rsh)  [default = 'ssh']
    options: remote service options (i.e. -v, -N, -L)  [default = '']
    background: run in background  [default = False]
    decode: ensure response is 'ascii'  [default = True]
    stdin: file-like object to serve as standard input for the remote process
        '''
        if self.message is None:
            self.message = 'echo %s' % self.name #' '?
        else: # pare back down to 'command' # better, just save _command?
            if self.launcher:
                self.message = self.message.split(self.launcher, 1)[-1]
            if self.options:
                self.message = self.message.split(self.options, 1)[-1]
            if self.host:
                self.message = self.message.split(self.host, 1)[-1].strip()
            quote = ('"',"'")
            if self.message.startswith(quote) or self.message.endswith(quote):
                self.message = self.message[1:-1]
        if self.stdin is None:
            import sys
            self.stdin = sys.stdin
        for key, value in kwds.items():
            if key == 'command':
                self.message = value
            elif key == 'host':
                self.host = value
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
        self.message = '%s %s %s "%s"' % (self.launcher,
                                          self.options,
                                          self.host,
                                          self.message)
        names=['message','host','launcher','options','background','stdin','codec']
        return dict((i,getattr(self, i)) for i in names)

    # interface
    __call__ = config
    pass


if __name__ == '__main__':
    pass


# End of file
