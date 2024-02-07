#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
#
# adapted from J. Kim & M. McKerns utility functions
"""
utilities for distributed computing
"""

import os

def _str(byte, codec=None):
    """convert bytes to string using the given codec (default is 'ascii')"""
    if codec is False or not hasattr(byte, 'decode'): return byte
    return byte.decode(codec or 'ascii')

def _b(string, codec=None):
    """convert string to bytes using the given codec (default is 'ascii')"""
    if codec is False or not hasattr(string, 'encode'): return string
    return string.encode(codec or 'ascii')

def print_exc_info():
    """thread-safe return of string from print_exception call"""

    import traceback
    import io
    
    sio = io.StringIO()
    traceback.print_exc(file=sio) #thread-safe print_exception to string
    sio.seek(0, 0)
    
    return sio.read()


def spawn(onParent, onChild):
    """a unidirectional fork wrapper

Calls onParent(pid, fromchild) in parent process,
      onChild(pid, toparent) in child process.
    """
    c2pread, c2pwrite = os.pipe()
        
    pid = os.fork()
    if pid > 0:
        os.close(c2pwrite)
        fromchild = os.fdopen(c2pread, 'rb')
        return onParent(pid, fromchild)

    os.close(c2pread)
    toparent = os.fdopen(c2pwrite, 'wb', 0)
    pid = os.getpid()

    return onChild(pid, toparent)


def spawn2(onParent, onChild):
    """a bidirectional fork wrapper

Calls onParent(pid, fromchild, tochild) in parent process,
      onChild(pid, fromparent, toparent) in child process.
    """
    p2cread, p2cwrite = os.pipe()
    c2pread, c2pwrite = os.pipe()
        
    pid = os.fork()
    if pid > 0:
        os.close(p2cread)
        os.close(c2pwrite)
        fromchild = os.fdopen(c2pread, 'rb')
        tochild = os.fdopen(p2cwrite, 'wb', 0)
        return onParent(pid, fromchild, tochild)

    os.close(p2cwrite)
    os.close(c2pread)
    fromparent = os.fdopen(p2cread, 'rb')
    toparent = os.fdopen(c2pwrite, 'wb', 0)
    pid = os.getpid()

    return onChild(pid, fromparent, toparent)


# End of file
