#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
"""
This module contains the base classes for pathos pool and pipe objects,
and describes the map and pipe interfaces.  A pipe is defined as a
connection between two 'nodes', where a node is something that does
work.  A pipe may be a one-way or two-way connection.  A map is defined
as a one-to-many connection between nodes.  In both map and pipe
connections, results from the connected nodes can be returned to the
calling node.  There are several variants of pipe and map, such as
whether the connection is blocking, or ordered, or asynchronous.  For
pipes, derived methods must overwrite the 'pipe' method, while maps
must overwrite the 'map' method.  Pipes and maps are available from
worker pool objects, where the work is done by any of the workers
in the pool.  For more specific point-to-point connections (such as
a pipe between two specific compute nodes), use the pipe object
directly.


Usage
=====

A typical call to a pathos map will roughly follow this example:

    >>> # instantiate and configure the worker pool
    >>> from pathos.pools import ProcessPool
    >>> pool = ProcessPool(nodes=4)
    >>>
    >>> # do a blocking map on the chosen function
    >>> results = pool.map(pow, [1,2,3,4], [5,6,7,8])
    >>>
    >>> # do a non-blocking map, then extract the results from the iterator
    >>> results = pool.imap(pow, [1,2,3,4], [5,6,7,8])
    >>> print("...")
    >>> results = list(results)
    >>>
    >>> # do an asynchronous map, then get the results
    >>> results = pool.amap(pow, [1,2,3,4], [5,6,7,8])
    >>> while not results.ready():
    ...     time.sleep(5); print(".", end=' ')
    ...
    >>> results = results.get()


Notes
=====

Each of the pathos worker pools rely on a different transport protocol
(e.g. threads, multiprocessing, etc), where the use of each pool comes
with a few caveats.  See the usage documentation and examples for each
worker pool for more information.

"""
__all__ = ['AbstractPipeConnection', 'AbstractWorkerPool']

class AbstractPipeConnection(object):
    """
AbstractPipeConnection base class for pathos pipes.
    """
    def __init__(self, *args, **kwds):
        """
Required input:
    ???

Additional inputs:
    ???

Important class members:
    ???

Other class members:
    ???
        """
        object.__init__(self)#, *args, **kwds)
        return
    def __repr__(self):
        return "<pipe %s>" % self.__class__.__name__
    # interface
    pass


class AbstractWorkerPool(object): # base for worker pool strategy or all maps?
    """
AbstractWorkerPool base class for pathos pools.
    """
    __nodes = 1
    def __init__(self, *args, **kwds):
        """
Important class members:
    nodes	- number (and potentially description) of workers
    ncpus       - number of worker processors
    servers     - list of worker servers
    scheduler   - the associated scheduler
    workdir     - associated $WORKDIR for scratch calculations/files

Other class members:
    scatter     - True, if uses 'scatter-gather' (instead of 'worker-pool')
    source      - False, if minimal use of TemporaryFiles is desired
    timeout	- number of seconds to wait for return value from scheduler
        """
        object.__init__(self)#, *args, **kwds)
        self.__init(*args, **kwds)
        self._id = kwds.get('id', None)
        return
    def __enter__(self):
        return self
    def __exit__(self, *args):
        #self.clear()
        return
    def __init(self, *args, **kwds):
        """default filter for __init__ inputs
        """
        # allow default arg for 'nodes', but not if in kwds
        if len(args):
            try:
                nodes = kwds['nodes']
                msg = "got multiple values for keyword argument 'nodes'"
                raise TypeError(msg)
            except KeyError:
                nodes = args[0]
        else: nodes = kwds.get('nodes', self.__nodes)
        try: self.nodes = nodes
        except TypeError: pass  # then self.nodes is read-only
        return
    def __map(self, f, *args, **kwds):
        """default filter for map inputs
        """
        # barf if given keywords
        if kwds:
            pass
       #    raise TypeError("map() takes no keyword arguments")
           #raise TypeError("'%s' is an invalid keyword for this function" % kwds.keys()[0])
        # at least one argument is required
        try:
            argz = [args[0]]
        except IndexError:
            raise TypeError("map() requires at least two args")
        return
    def __imap(self, f, *args, **kwds):
        """default filter for imap inputs
        """
        # barf if given keywords
        if kwds:
            pass
       #    raise TypeError("map() does not take keyword arguments")
           #raise TypeError("'%s' is an invalid keyword for this function" % kwds.keys()[0])
        # at least one argument is required
        try:
            argz = [args[0]]
        except IndexError:
            raise TypeError("imap() must have at least two arguments")
        return
    def __pipe(self, f, *args, **kwds):  #FIXME: need to think about this...
        """default filter for pipe inputs
        """
        # barf if given keywords
        if kwds:
            pass
       #    raise TypeError("pipe() does not take keyword arguments")
           #raise TypeError("'%s' is an invalid keyword for this function" % kwds.keys()[0])
        # a valid number of arguments are required
        try:
            vars = f.__code__.co_argcount
            defs = len(f.__defaults__)
            arglen = len(args)
            minlen = vars - defs
            if vars == minlen and arglen != vars: #XXX: argument vs arguments
              raise TypeError("%s() takes at exactly %s arguments (%s given)" % (f.__name__(), str(vars), str(arglen)))
            elif arglen > vars:
              raise TypeError("%s() takes at most %s arguments (%s given)" % (f.__name__(), str(vars), str(arglen)))
            elif arglen < (vars - defs):
              raise TypeError("%s() takes at least %s arguments (%s given)" % (f.__name__(), str(vars - defs), str(arglen)))
        except:
            pass
        return
    def _serve(self, *args, **kwds):
        """Create a new server if one isn't already initialized"""
        raise NotImplementedError
       #_pool = None
       #return _pool
    def clear(self):
        """Remove server with matching state"""
        raise NotImplementedError
       #return #XXX: return _pool? (i.e. pop)
    def map(self, f, *args, **kwds):
        """run a batch of jobs with a blocking and ordered map

Returns a list of results of applying the function f to the items of
the argument sequence(s). If more than one sequence is given, the
function is called with an argument list consisting of the corresponding
item of each sequence. Some maps accept the `chunksize` keyword, which
causes the sequence to be split into tasks of approximately the given size.
        """
       #self.__map(f, *args, **kwds)
        raise NotImplementedError
    def imap(self, f, *args, **kwds):
        """run a batch of jobs with a non-blocking and ordered map

Returns a list iterator of results of applying the function f to the items
of the argument sequence(s). If more than one sequence is given, the
function is called with an argument list consisting of the corresponding
item of each sequence. Some maps accept the `chunksize` keyword, which
causes the sequence to be split into tasks of approximately the given size.
        """
       #self.__imap(f, *args, **kwds)
        raise NotImplementedError
    def uimap(self, f, *args, **kwds):
        """run a batch of jobs with a non-blocking and unordered map

Returns a list iterator of results of applying the function f to the items
of the argument sequence(s). If more than one sequence is given, the
function is called with an argument list consisting of the corresponding
item of each sequence. The order of the resulting sequence is not guaranteed.
Some maps accept the `chunksize` keyword, which causes the sequence to be
split into tasks of approximately the given size.
        """
       #self.__imap(f, *args, **kwds)
        raise NotImplementedError
    def amap(self, f, *args, **kwds):
        """run a batch of jobs with an asynchronous map

Returns a results object which containts the results of applying the
function f to the items of the argument sequence(s). If more than one
sequence is given, the function is called with an argument list consisting
of the corresponding item of each sequence. To retrieve the results, call
the get() method on the returned results object. The call to get() is
blocking, until all results are retrieved. Use the ready() method on the
result object to check if all results are ready. Some maps accept the
`chunksize` keyword, which causes the sequence to be split into tasks of
approximately the given size.
        """
       #self.__map(f, *args, **kwds)
        raise NotImplementedError
    ########################################################################
    # PIPES
    def pipe(self, f, *args, **kwds):
        """submit a job and block until results are available

Returns result of calling the function f on a selected worker.  This function
will block until results are available.
        """
       #self.__pipe(f, *args, **kwds)
        raise NotImplementedError
    def apipe(self, f, *args, **kwds): # register a callback ?
        """submit a job asynchronously to a queue

Returns a results object which containts the result of calling the
function f on a selected worker. To retrieve the results, call the
get() method on the returned results object. The call to get() is
blocking, until the result is available. Use the ready() method on the
results object to check if the result is ready.
        """
       #self.__pipe(f, *args, **kwds)
        raise NotImplementedError
    ########################################################################
    def __repr__(self):
        return "<pool %s()>" % self.__class__.__name__
    def __get_nodes(self):
        """get the number of nodes in the pool"""
        return self.__nodes
    def __set_nodes(self, nodes):
        """set the number of nodes in the pool"""
        raise TypeError("nodes is a read-only attribute")
    # interface
    pass


