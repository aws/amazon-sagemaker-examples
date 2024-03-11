#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
"""
This module contains map and pipe interfaces to standard (i.e. serial) python.

Pipe methods provided:
    pipe        - blocking communication pipe             [returns: value]

Map methods provided:
    map         - blocking and ordered worker pool      [returns: list]
    imap        - non-blocking and ordered worker pool  [returns: iterator]


Usage
=====

A typical call to a pathos python map will roughly follow this example:

    >>> # instantiate and configure the worker pool
    >>> from pathos.serial import SerialPool
    >>> pool = SerialPool()
    >>>
    >>> # do a blocking map on the chosen function
    >>> print(pool.map(pow, [1,2,3,4], [5,6,7,8]))
    >>>
    >>> # do a non-blocking map, then extract the results from the iterator
    >>> results = pool.imap(pow, [1,2,3,4], [5,6,7,8])
    >>> print("...")
    >>> print(list(results))
    >>>
    >>> # do one item at a time, using a pipe
    >>> print(pool.pipe(pow, 1, 5))
    >>> print(pool.pipe(pow, 2, 6))


Notes
=====

This worker pool leverages the built-in python maps, and thus does not have
limitations due to serialization of the function f or the sequences in args.
The maps in this worker pool have full functionality whether run from a script
or in the python interpreter, and work reliably for both imported and
interactively-defined functions.

"""
__all__ = ['SerialPool']

from pathos.abstract_launcher import AbstractWorkerPool
__get_nodes__ = AbstractWorkerPool._AbstractWorkerPool__get_nodes
__set_nodes__ = AbstractWorkerPool._AbstractWorkerPool__set_nodes

from builtins import map as _map
_apply = lambda f, args, kwds: f(*args, **kwds)
_imap = _map

#XXX: good for interface... or bad idea?
__STATE = _SerialPool__STATE = {}

#FIXME: in python3.x mp.map returns a list, mp.imap an iterator
class SerialPool(AbstractWorkerPool):
    """
Mapper that leverages standard (i.e. serial) python maps.
    """
    # interface (no __init__)
    _exiting = False

    def map(self, f, *args, **kwds):
       #AbstractWorkerPool._AbstractWorkerPool__map(self, f, *args, **kwds)
        if self._exiting: self._is_alive()
        return _map(f, *args)#, **kwds) # chunksize
    map.__doc__ = AbstractWorkerPool.map.__doc__
    def imap(self, f, *args, **kwds):
       #AbstractWorkerPool._AbstractWorkerPool__imap(self, f, *args, **kwds)
        if self._exiting: self._is_alive()
        return _imap(f, *args)#, **kwds) # chunksize
    imap.__doc__ = AbstractWorkerPool.imap.__doc__
    ########################################################################
    # PIPES
    def pipe(self, f, *args, **kwds):
       #AbstractWorkerPool._AbstractWorkerPool__pipe(self, f, *args, **kwds)
        if self._exiting: self._is_alive()
        return _apply(f, args, kwds)
    pipe.__doc__ = AbstractWorkerPool.pipe.__doc__
    #XXX: generator/yield provides simple ipipe? apipe? what about coroutines?
    ########################################################################
    def restart(self, force=False):
        "restart a closed pool"
        if not force and not self._exiting: self._is_alive(negate=True)
        # 'clear' and 'serve'
        if self._exiting: # i.e. is destroyed
            self.clear()
        return
    def _is_alive(self, negate=False, run=True):
        RUN,CLOSE,TERMINATE = 0,1,2
        pool = lambda :None
        pool._state = RUN if negate else CLOSE
        if negate and run: # throw error if alive (exiting=True)
            assert pool._state != RUN
        elif negate: # throw error if alive (exiting=True)
            assert pool._state in (CLOSE, TERMINATE)
        else:     # throw error if not alive (exiting=False)
            raise ValueError("Pool not running")
    def close(self):
        "close the pool to any new jobs"
        self._exiting = True
        return
    def terminate(self):
        "a more abrupt close"
        self.close()
        self.join()
        return
    def join(self):
        "cleanup the closed worker processes"
        if not self._exiting: 
            self._is_alive(negate=True, run=False)
        self._exiting = True
        return
    def clear(self):
        """hard restart"""
        self._exiting = False
        return
    ########################################################################
    # interface
    __get_nodes = __get_nodes__
    __set_nodes = __set_nodes__
    nodes = property(__get_nodes, __set_nodes)
    __state__ = __STATE
    pass


# backward compatibility
PythonSerial = SerialPool

# EOF
