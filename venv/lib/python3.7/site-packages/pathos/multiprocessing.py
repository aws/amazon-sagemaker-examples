#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
"""
This module contains map and pipe interfaces to python's multiprocessing module.

Pipe methods provided:
    pipe        - blocking communication pipe             [returns: value]
    apipe       - asynchronous communication pipe         [returns: object]

Map methods provided:
    map         - blocking and ordered worker pool        [returns: list]
    imap        - non-blocking and ordered worker pool    [returns: iterator]
    uimap       - non-blocking and unordered worker pool  [returns: iterator]
    amap        - asynchronous worker pool                [returns: object]


Usage
=====

A typical call to a pathos multiprocessing map will roughly follow this example:

    >>> # instantiate and configure the worker pool
    >>> from pathos.multiprocessing import ProcessPool
    >>> pool = ProcessPool(nodes=4)
    >>>
    >>> # do a blocking map on the chosen function
    >>> print(pool.map(pow, [1,2,3,4], [5,6,7,8]))
    >>>
    >>> # do a non-blocking map, then extract the results from the iterator
    >>> results = pool.imap(pow, [1,2,3,4], [5,6,7,8])
    >>> print("...")
    >>> print(list(results))
    >>>
    >>> # do an asynchronous map, then get the results
    >>> results = pool.amap(pow, [1,2,3,4], [5,6,7,8])
    >>> while not results.ready():
    ...     time.sleep(5); print(".", end=' ')
    ...
    >>> print(results.get())
    >>>
    >>> # do one item at a time, using a pipe
    >>> print(pool.pipe(pow, 1, 5))
    >>> print(pool.pipe(pow, 2, 6))
    >>>
    >>> # do one item at a time, using an asynchronous pipe
    >>> result1 = pool.apipe(pow, 1, 5)
    >>> result2 = pool.apipe(pow, 2, 6)
    >>> print(result1.get())
    >>> print(result2.get())


Notes
=====

This worker pool leverages the python's multiprocessing module, and thus
has many of the limitations associated with that module. The function f and
the sequences in args must be serializable. The maps in this worker pool
have full functionality whether run from a script or in the python
interpreter, and work reliably for both imported and interactively-defined
functions. Unlike python's multiprocessing module, pathos.multiprocessing maps
can directly utilize functions that require multiple arguments.

"""
__all__ = ['ProcessPool','_ProcessPool']

#FIXME: probably not good enough... should store each instance with a uid
__STATE = _ProcessPool__STATE = {}

from pathos.abstract_launcher import AbstractWorkerPool
from pathos.helpers.mp_helper import starargs as star
from pathos.helpers import cpu_count, freeze_support, ProcessPool as Pool
import warnings
import sys
OLD312a7 = (sys.hexversion < 0x30c00a7)

# 'forward' compatibility
_ProcessPool = Pool

class ProcessPool(AbstractWorkerPool):
    """
Mapper that leverages python's multiprocessing.
    """
    def __init__(self, *args, **kwds):
        """\nNOTE: if number of nodes is not given, will autodetect processors.
\nNOTE: additional keyword input is optional, with:
    id          - identifier for the pool
    initializer - function that takes no input, called when node is spawned
    initargs    - tuple of args for initializers that have args
    maxtasksperchild - int that limits the max number of tasks per node
        """
        hasnodes = 'nodes' in kwds; arglen = len(args)
        if 'ncpus' in kwds and (hasnodes or arglen):
            msg = "got multiple values for keyword argument 'ncpus'"
            raise TypeError(msg)
        elif hasnodes: #XXX: multiple try/except is faster?
            if arglen:
                msg = "got multiple values for keyword argument 'nodes'"
                raise TypeError(msg)
            kwds['ncpus'] = kwds.pop('nodes')
        elif arglen:
            kwds['ncpus'] = args[0]
        if 'processes' in kwds:
            if 'ncpus' in kwds:
                msg = "got multiple values for keyword argument 'processes'"
                raise TypeError(msg)
            kwds['ncpus'] = kwds.pop('processes')
        self.__nodes = kwds.pop('ncpus', cpu_count())

        # Create an identifier for the pool
        self._id = kwds.pop('id', None) #'pool'
        if self._id is None:
            self._id = self.__nodes

        self._kwds = kwds

        # Create a new server if one isn't already initialized
        self._serve()
        return
    if AbstractWorkerPool.__init__.__doc__: __init__.__doc__ = AbstractWorkerPool.__init__.__doc__ + __init__.__doc__
   #def __exit__(self, *args):
   #    self._clear()
   #    return
    def _serve(self, nodes=None): #XXX: should be STATE method; use id
        """Create a new server if one isn't already initialized"""
        if nodes is None: nodes = self.__nodes
        _pool = __STATE.get(self._id, None)
        if not _pool or nodes != _pool.__nodes or self._kwds != _pool._kwds:
            self._clear()
            _pool = Pool(nodes, **self._kwds)
            _pool.__nodes = nodes
            _pool._kwds = self._kwds
            __STATE[self._id] = _pool
        return _pool
    def _clear(self): #XXX: should be STATE method; use id
        """Remove server with matching state"""
        _pool = __STATE.get(self._id, None)
        if _pool and self.__nodes == _pool.__nodes and self._kwds == _pool._kwds:
            _pool.close()
            _pool.join()
            __STATE.pop(self._id, None)
        return #XXX: return _pool?
    clear = _clear
    def map(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__map(self, f, *args, **kwds)
        _pool = self._serve()
        with warnings.catch_warnings():
            if not OLD312a7:
                warnings.filterwarnings('ignore', category=DeprecationWarning)
            return _pool.map(star(f), zip(*args), **kwds)
    map.__doc__ = AbstractWorkerPool.map.__doc__
    def imap(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__imap(self, f, *args, **kwds)
        _pool = self._serve()
        with warnings.catch_warnings():
            if not OLD312a7:
                warnings.filterwarnings('ignore', category=DeprecationWarning)
            return _pool.imap(star(f), zip(*args), **kwds)
    imap.__doc__ = AbstractWorkerPool.imap.__doc__
    def uimap(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__imap(self, f, *args, **kwds)
        _pool = self._serve()
        with warnings.catch_warnings():
            if not OLD312a7:
                warnings.filterwarnings('ignore', category=DeprecationWarning)
            return _pool.imap_unordered(star(f), zip(*args), **kwds)
    uimap.__doc__ = AbstractWorkerPool.uimap.__doc__
    def amap(self, f, *args, **kwds): # register a callback ?
        AbstractWorkerPool._AbstractWorkerPool__map(self, f, *args, **kwds)
        _pool = self._serve()
        with warnings.catch_warnings():
            if not OLD312a7:
                warnings.filterwarnings('ignore', category=DeprecationWarning)
            return _pool.map_async(star(f), zip(*args), **kwds)
    amap.__doc__ = AbstractWorkerPool.amap.__doc__
    ########################################################################
    # PIPES
    def pipe(self, f, *args, **kwds):
       #AbstractWorkerPool._AbstractWorkerPool__pipe(self, f, *args, **kwds)
        _pool = self._serve()
        with warnings.catch_warnings():
            if not OLD312a7:
                warnings.filterwarnings('ignore', category=DeprecationWarning)
            return _pool.apply(f, args, kwds)
    pipe.__doc__ = AbstractWorkerPool.pipe.__doc__
    def apipe(self, f, *args, **kwds): # register a callback ?
       #AbstractWorkerPool._AbstractWorkerPool__apipe(self, f, *args, **kwds)
        _pool = self._serve()
        with warnings.catch_warnings():
            if not OLD312a7:
                warnings.filterwarnings('ignore', category=DeprecationWarning)
            return _pool.apply_async(f, args, kwds)
    apipe.__doc__ = AbstractWorkerPool.apipe.__doc__
    ########################################################################
    def __repr__(self):
        mapargs = (self.__class__.__name__, self.ncpus)
        return "<pool %s(ncpus=%s)>" % mapargs
    def __get_nodes(self):
        """get the number of nodes used in the map"""
        return self.__nodes
    def __set_nodes(self, nodes):
        """set the number of nodes used in the map"""
        self._serve(nodes)
        self.__nodes = nodes
        return
    ########################################################################
    def restart(self, force=False):
        "restart a closed pool"
        _pool = __STATE.get(self._id, None)
        if _pool and self.__nodes == _pool.__nodes and self._kwds == _pool._kwds:
            RUN = 0
            if not force:
                assert _pool._state != RUN
            # essentially, 'clear' and 'serve'
            self._clear()
            _pool = Pool(self.__nodes, **self._kwds)
            _pool.__nodes = self.__nodes
            _pool._kwds = self._kwds
            __STATE[self._id] = _pool
        return _pool
    def close(self):
        "close the pool to any new jobs"
        _pool = __STATE.get(self._id, None)
        if _pool and self.__nodes == _pool.__nodes:
            _pool.close()
        return
    def terminate(self):
        "a more abrupt close"
        _pool = __STATE.get(self._id, None)
        if _pool and self.__nodes == _pool.__nodes:
            _pool.terminate()
        return
    def join(self):
        "cleanup the closed worker processes"
        _pool = __STATE.get(self._id, None)
        if _pool and self.__nodes == _pool.__nodes:
            _pool.join()
        return
    # interface
    ncpus = property(__get_nodes, __set_nodes)
    nodes = property(__get_nodes, __set_nodes)
    __state__ = __STATE
    pass


# backward compatibility
from pathos.helpers import ThreadPool
from pathos.threading import ThreadPool as ThreadingPool
ProcessingPool = ProcessPool

# EOF
