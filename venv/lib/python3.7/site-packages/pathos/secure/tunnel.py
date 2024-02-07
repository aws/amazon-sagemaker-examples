#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
#
# adapted from J. Kim & M. McKerns' Tunnel class
"""
This module contains the base class for secure tunnel connections, and
describes the pathos tunnel interface.  See the following for an example.


Usage
=====

A typical call to a pathos 'tunnel' will roughly follow this example:

    >>> # instantiate the tunnel, providing it with a unique identifier
    >>> tunnel = Tunnel('tunnel')
    >>>
    >>> # establish a tunnel to the remote host and port
    >>> remotehost = 'remote.host.edu'
    >>> remoteport = 12345
    >>> localport = tunnel.connect(remotehost, remoteport)
    >>> print("Tunnel connected at local port: %s" % tunnel._lport)
    >>>
    >>> # pause script execution to maintain the tunnel (i.e. do something)
    >>> sys.stdin.readline()
    >>>
    >>> # tear-down the tunneled connection
    >>> tunnel.disconnect()
 
"""
__all__ = ['Tunnel','TunnelException']

import os
import signal
import random
import string
from pathos.secure import Pipe

class TunnelException(Exception):
    '''Exception for failure to establish ssh tunnel'''
    pass

class Tunnel(object):
    """a ssh-tunnel launcher for parallel and distributed computing."""
    #MINPORT = 49152    
    MINPORT = 1024 
    MAXPORT = 65535
    verbose = True

    def connect(self, host, port=None, through=None):
        '''establish a secure shell tunnel between local and remote host

Input:
    host     -- remote hostname  [user@host:path is also valid]
    port     -- remote port number

Additional Input:
    through  -- 'tunnel-through' hostname  [default = None]
        '''
        from pathos.portpicker import portnumber
        if port is None:
            from pathos.core import randomport
            port = randomport(through) if through else randomport(host)

        pick = portnumber(self.MINPORT, self.MAXPORT)
        while True:
            localport = pick()
            if localport < 0:
                raise TunnelException('No available local port')
            #print('Trying port %d...' % localport)
            
            try:
                self._connect(localport, host, port, through=through)
                #print('SSH tunnel %d:%s:%d' % (localport, host, port))
            except TunnelException as e: # breaks 2.5 compatibility
                if e.args[0] == 'bind':
                    self.disconnect()
                    continue
                else:
                    self.__disconnect()
                    raise TunnelException('Connection failed')
                
            self.connected = True
            return localport

    def disconnect(self):
        '''destroy the ssh tunnel'''
        #FIXME: grep (?) for self._launcher.message, then kill the pid
        if self._pid > 0:
            if self.verbose: print('Kill ssh pid=%d' % self._pid)
            os.kill(self._pid, signal.SIGTERM)
            os.waitpid(self._pid, 0)
            self.__disconnect()
        return

    def __disconnect(self):
        '''disconnect tunnel internals'''
        self._pid = 0
        self.connected = False
        self._lport = None
        self._rport = None
        self._host = None
        return

    def __init__(self, name=None, **kwds):
        '''create a ssh tunnel launcher

Inputs:
    name        -- a unique identifier (string) for the launcher
        '''
        xyz = string.ascii_letters
        self.name = ''.join(random.choice(xyz) for i in range(16)) \
               if name is None else name
        self._launcher = Pipe('launcher')

        self.__disconnect()
        if kwds: self.connect(**kwds)
        return

    def __repr__(self):
        if not self.connected:
            return "Tunnel('%s')" % self.name
        try:
            msg = self._launcher.message.split(' ',1)[-1].rstrip('"').rstrip()
        except: msg = self._launcher.message
        return "Tunnel('%s')" % msg

    def _connect(self, localport, remotehost, remoteport, through=None):
        options = '-q -N -L %d:%s:%d' % (localport, remotehost, remoteport)
        command = ''
        if through: rhost = through
        else: rhost = remotehost
        self._launcher(host=rhost, command=command,
                       options=options, background=True) #XXX: MMM
                      #options=options, background=False)
        self._launcher.launch()
        self._lport = localport
        self._rport = remoteport
        self._host = rhost
        self._pid = self._launcher.pid() #FIXME: should be tunnel_pid [pid()+1]
        line = self._launcher.response()
        if line:
            if line.startswith('bind'):
                raise TunnelException('bind')
            else:
                print(line)
                raise TunnelException('failure')
        return

if __name__ == '__main__':
    pass


# End of file
