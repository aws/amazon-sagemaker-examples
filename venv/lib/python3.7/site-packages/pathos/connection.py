#!/usr/bin/env python
#
# Originally from pythia-0.8 pyre.mpi.Launcher.py (svn:danse.us/pyre -r2)
# Forked by: Mike McKerns (January 2004)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2004-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
"""
This module contains the base class for popen pipes, and describes
the popen pipe interface. The 'config' method can be overwritten
for pipe customization. The pipe's 'launch' method can be overwritten
with a derived pipe's new execution algorithm. See the following for
an example of standard use.


Usage
=====

A typical call to a popen 'pipe' will roughly follow this example:

    >>> # instantiate the pipe
    >>> pipe = Pipe()
    >>>
    >>> # configure the pipe to stage the command
    >>> pipe(command='hostname')
    >>>
    >>> # execute the launch and retrieve the response
    >>> pipe.launch()
    >>> print(pipe.response())
 
"""
__all__ = ['Pipe', 'PipeException']

import os
import sys
import signal
import random
import string
from pathos.selector import Selector
from pathos.util import _str

class PipeException(Exception):
    '''Exception for failure to launch a command'''
    pass

# broke backward compatability: 30/05/14 ==> replace base-class almost entirely
class Pipe(object):
    """a popen-based pipe for parallel and distributed computing"""
    verbose = True
    from pathos import logger
    _debug = logger(level=30) # logging.WARN
    del logger

    def __init__(self, name=None, **kwds):
        """create a popen-pipe

Inputs:
    name: a unique identifier (string) for the pipe
    command: a command to send  [default = 'echo <name>']
    background: run in background  [default = False]
    decode: ensure response is 'ascii'  [default = True]
    stdin: file-like object to serve as standard input for the remote process
        """
        xyz = string.ascii_letters
        self.name = ''.join(random.choice(xyz) for i in range(16)) \
               if name is None else name

        self.background = kwds.pop('background', False)
        self.stdin = kwds.pop('stdin', sys.stdin)
        self.codec = kwds.pop('decode', 'ascii')
        self.message = kwds.pop('command', 'echo %s' % self.name) #' '?
        self._response = None
        self._pid = 0
        self.config(**kwds)
        return

    def __repr__(self):
        return "Pipe('%s')" % self.message

    def config(self, **kwds):
        '''configure the pipe using given keywords

(Re)configure the pipe for the following inputs:
    command: a command to send  [default = 'echo <name>']
    background: run in background  [default = False]
    decode: ensure response is 'ascii'  [default = True]
    stdin: file-like object to serve as standard input for the remote process
        '''
        if self.message is None:
            self.message = 'echo %s' % self.name  #' '?
        if self.stdin is None:
            self.stdin = sys.stdin
        if self.codec is None:
            self.codec = 'ascii'
        for key, value in kwds.items():
            if key == 'command':
                self.message = value
            elif key == 'background':
                self.background = value
            elif key == 'decode':
                self.codec = value
            elif key == 'stdin':
                self.stdin = value

        self._stdout = None
        names=['message','background','stdin','codec']
        return dict((i,getattr(self, i)) for i in names)

    def launch(self):
        '''launch a configured command'''
        self._response = None
        self._execute()  # preempt with pox.which(message.split()[0]) ?
        return

    def _execute(self):
       #'''execute by piping the command, & saving the file object'''
        from subprocess import Popen, PIPE, STDOUT
        #XXX: what if saved list/dict of _stdout instead of just the one?
        #     could associated name/_pid and _stdout
        if self.background: #Spawn a background process 
            try:
                p = Popen(self.message, shell=True,
                          stdin=self.stdin, stdout=PIPE,
                          stderr=STDOUT, close_fds=True)
            except:
                raise PipeException('failure to pipe: %s' % self.message)
            self._pid = p.pid #get fileobject pid
            self._stdout = p.stdout #save fileobject
        else:
            try:
                p = Popen(self.message, shell=True,
                          stdin=self.stdin, stdout=PIPE)
            except:
                raise PipeException('failure to pipe: %s' % self.message)
            self._stdout = p.stdout
            self._pid = 0 #XXX: MMM --> or -1 ?
        return

    def response(self):
        '''Return the response from the launched process.
        Return None if no response was received yet from a background process.
        ''' #XXX: return bytes, decode to ascii, take encoding, or ??? 

        if self._stdout is None:
            raise PipeException("'launch' is required after any reconfiguration")
        if self.codec is True: codec = 'ascii'
        elif self.codec is False: codec = False
        elif self.codec is None: codec = False
        else: codec = self.codec
        if self._response is not None: return _str(self._response, codec)

        # when running in foreground _pid is 0 (may change to -1)
        if self._pid <= 0:
            self._response = self._stdout.read()
            return _str(self._response, codec)
        
        # handle response from a background process
        def onData(selector, fobj):
            if self.verbose: print("handling pipe response")
            self._debug.info('on_remote')
            self._response = fobj.read()
            selector.state = False
            return

        def onTimeout(selector):
            selector.state = False
        
        sel = Selector()
        #sel._info.activate()
        sel.notifyOnReadReady(self._stdout, onData)
        sel.notifyWhenIdle(onTimeout)
        sel.watch(2.0)
        # reset _response to None to allow capture of a next response
        # from a background process
        return _str(self._response, codec)

    def pid(self):
        '''get pipe pid'''
        return self._pid

    def kill(self):
        '''terminate the pipe'''
        if self._pid > 0:
            if self.verbose: print('Kill pid=%d' % self._pid)
            os.kill(self._pid, signal.SIGTERM)
            os.waitpid(self._pid, 0)
            self._pid = 0
        return

    # interface
    __call__ = config
    pass


# End of file 
