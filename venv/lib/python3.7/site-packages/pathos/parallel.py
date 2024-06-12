#!/usr/bin/env python
#
# Based on code by Kirk Strauser <kirk@strauser.com>
# Rev: 1139; Date: 2008-04-16
# (see license text in pathos.pp_map)
#
# Forked by: Mike McKerns (April 2008)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
#
# Modified to meet the pathos pool API
"""
This module contains map and pipe interfaces to the parallelpython (pp) module.

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

A typical call to a pathos pp map will roughly follow this example:

    >>> # instantiate and configure the worker pool
    >>> from pathos.pp import ParallelPool
    >>> pool = ParallelPool(nodes=4)
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

This worker pool leverages the parallelpython (pp) module, and thus
has many of the limitations associated with that module. The function f and
the sequences in args must be serializable. The maps in this worker pool
have full functionality when run from a script, but may be somewhat limited
when used in the python interpreter. Both imported and interactively-defined
functions in the interpreter session may fail due to the pool failing to
find the source code for the target function. For a work-around, try:

    >>> # instantiate and configure the worker pool
    >>> from pathos.pp import ParallelPool
    >>> pool = ParallelPool(nodes=4)
    >>>
    >>> # wrap the function, so it can be used interactively by the pool
    >>> def wrapsin(*args, **kwds):
    >>>      from math import sin
    >>>      return sin(*args, **kwds)
    >>>
    >>> # do a blocking map using the wrapped function
    >>> results = pool.map(wrapsin, [1,2,3,4,5])

"""
__all__ = ['ParallelPool', 'stats']

from pathos.helpers import parallelpython as pp
from pathos.helpers import cpu_count

import builtins

#FIXME: probably not good enough... should store each instance with a uid
__STATE = _ParallelPool__STATE = {}

def __print_stats(servers=None):
    "print stats from the pp.Server"
    FROM_STATE = True
    if servers is None: servers = list(__STATE.values())
    else: FROM_STATE = False
    try:
        servers = tuple(servers)
    except TypeError:
        servers = (servers,)
    if not servers:
        msg = '; no active' if FROM_STATE else ' for the requested'
        print("Stats are not available%s servers.\n" % msg)
        return
    for server in servers: # fails if not pp.Servers
        #XXX: also print ids? (__STATE.keys())?
        server.print_stats()
    return

#XXX: better return object(?) to query? | is per run? compound?
def stats(pool=None):
    "return a string containing stats response from the pp.Server"
    server = None if pool is None else __STATE.get(pool._id, tuple())

    import io
    import sys
    stdout = sys.stdout
    try:
        sys.stdout = result = io.StringIO()
        __print_stats(server)
    except:
        result = None #XXX: better throw an error?
    sys.stdout = stdout
    result = result.getvalue() if result else ''
    return result


from pathos.abstract_launcher import AbstractWorkerPool
from pathos.helpers.pp_helper import ApplyResult, MapResult

#XXX: should look into parallelpython for 'cluster computing'
class ParallelPool(AbstractWorkerPool):
    """
Mapper that leverages parallelpython (i.e. pp) maps.
    """
    def __init__(self, *args, **kwds):
        """\nNOTE: if number of nodes is not given, will autodetect processors.
\nNOTE: if a tuple of servers is not provided, defaults to localhost only.
\nNOTE: additional keyword input is optional, with:
    id          - identifier for the pool
    servers     - tuple of pp.Servers
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
        self.__nodes = None
        self.__servers = ()

        ncpus = kwds.get('ncpus', None)
       #servers = kwds.get('servers', ('*',)) # autodetect
        servers = kwds.get('servers', ()) # only localhost
        if servers is None: servers = ()
       #from _ppserver_config import ppservers as servers # config file

        # Create an identifier for the pool
        self._id = kwds.get('id', None) #'server'
        if self._id is None:
            _nodes = str(ncpus) if type(ncpus) is int else '*'
            self._id = '@'.join([_nodes, '+'.join(sorted(servers))])

        #XXX: throws 'socket.error' when starting > 1 server with autodetect
        # Create a new server if one isn't already initialized
        # ...and set the requested level of multi-processing
        self._exiting = False
        _pool = self._serve(nodes=ncpus, servers=servers)
        #XXX: or register new UID for each instance?
        #_pool.set_ncpus(ncpus or 'autodetect') # no ncpus=0
       #print("configure %s local workers" % _pool.get_ncpus())
        return
    if AbstractWorkerPool.__init__.__doc__: __init__.__doc__ = AbstractWorkerPool.__init__.__doc__ + __init__.__doc__
   #def __exit__(self, *args):
   #    self._clear()
   #    return
    def _serve(self, nodes=None, servers=None): #XXX: is a STATE method; use id
        """Create a new server if one isn't already initialized""" 
        # get nodes and servers in form used by pp.Server
        if nodes is None: nodes = self.nodes #XXX: autodetect must be explicit
        if nodes in ['*']: nodes = 'autodetect'
        if servers is None:
            servers = tuple(sorted(self.__servers)) # no servers is ()
        elif servers in ['*', 'autodetect']: servers = ('*',)
        # if no server, create one
        _pool = __STATE.get(self._id, None)
        if not _pool:
            _pool = pp.Server(ppservers=servers)
        # convert to form returned by pp.Server, then compare
        _auto = [('*',)] if _pool.auto_ppservers else []
        _servers = sorted(_pool.ppservers + _auto)
        _servers = tuple(':'.join((str(i) for i in tup)) for tup in _servers)
        if servers != _servers: #XXX: assume servers specifies ports if desired
            _pool = pp.Server(ppservers=servers)
        # convert to form returned by pp.Server, then compare
        _nodes = cpu_count() if nodes=='autodetect' else nodes
        if _nodes != _pool.get_ncpus():
            _pool.set_ncpus(nodes) # allows ncpus=0
        # set (or 'repoint') the server
        __STATE[self._id] = _pool
        # set the 'self' internals
        self.__nodes = None if nodes in ['autodetect'] else nodes
        self.__servers = servers
        return _pool
    def _clear(self): #XXX: should be STATE method; use id
        """Remove server with matching state"""
        _pool = __STATE.get(self._id, None)
        if not self._equals(_pool):
            return
        # it's the 'same' (better to check _pool.secret?)
        _pool.destroy()
        __STATE.pop(self._id, None)
        self._exiting = False
        return #XXX: return _pool?
    clear = _clear
    def map(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__map(self, f, *args, **kwds)
        return list(self.imap(f, *args)) # chunksize
    map.__doc__ = AbstractWorkerPool.map.__doc__
    def imap(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__imap(self, f, *args, **kwds)
        def submit(*argz):
            """send a job to the server"""
            _pool = self._serve()
           #print("using %s local workers" % _pool.get_ncpus())
            try:
                return _pool.submit(f, argz, globals=globals())
            except pp.DestroyedServerError:
                self._is_alive(None)
        # submit all jobs, then collect results as they become available
        return (subproc() for subproc in list(builtins.map(submit, *args))) # chunksize
    imap.__doc__ = AbstractWorkerPool.imap.__doc__
    def uimap(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__imap(self, f, *args, **kwds)
        def submit(*argz):
            """send a job to the server"""
            _pool = self._serve()
           #print("using %s local workers" % _pool.get_ncpus())
            try:
                return _pool.submit(f, argz, globals=globals())
            except pp.DestroyedServerError:
                self._is_alive(None)
        def imap_unordered(it):
            """build a unordered map iterator"""
            it = list(it)
            while len(it):
                for i,job in enumerate(it):
                    if job.finished:
                        yield it.pop(i)()
                        break
                # yield it.pop(0).get()  # wait for the first element?
                # *subprocess*           # alternately, loop in a subprocess
            return #raise StopIteration
        # submit all jobs, then collect results as they become available
        return imap_unordered(builtins.map(submit, *args)) # chunksize
    uimap.__doc__ = AbstractWorkerPool.uimap.__doc__
    def amap(self, f, *args, **kwds):
        AbstractWorkerPool._AbstractWorkerPool__map(self, f, *args, **kwds)
        def submit(*argz):
            """send a job to the server"""
            _pool = self._serve()
           #print("using %s local workers" % _pool.get_ncpus())
            try:
                return _pool.submit(f, argz, globals=globals())
            except pp.DestroyedServerError:
                self._is_alive(None)
        override = True if 'size' in kwds else False
        elem_size = kwds.pop('size', 2)
        length = min(len(task) for task in args)
        args = zip(*args)  #XXX: zip iterator ok? or should be list?
        # submit all jobs, to be collected later with 'get()'
        tasks = [submit(*task) for task in args]
        tasks = [ApplyResult(task) for task in tasks]
        # build a correctly sized results object
        nodes = self.nodes
        if self.nodes in ['*','autodetect',None]:
            _pool = self._serve()
            nodes = _pool.get_ncpus() #NOTE: local only
            #nodes = _pool.get_active_nodes() #XXX: ppft version?
            #nodes = min(j for (i,j) in nodes.items() if i != 'local')
        if not nodes: nodes = 1
        # try to quickly find a small chunksize that gives good results
        maxsize = 2**62 #XXX: HOPEFULLY, this will never be reached...
        chunksize = 1 # chunksize
        while chunksize < maxsize:
            chunksize, extra = divmod(length, nodes * elem_size)
            if override: break # the user *wants* to override this loop
            if extra >= length: break # we found something that 'works'
            elem_size = elem_size * 2
        if extra: chunksize += 1
        m = MapResult((chunksize,length))
        # queue the tasks
        m.queue(*tasks)
        return m
    amap.__doc__ = AbstractWorkerPool.amap.__doc__
    ########################################################################
    # PIPES
    def pipe(self, f, *args, **kwds):
       #AbstractWorkerPool._AbstractWorkerPool__pipe(self, f, *args, **kwds)
        # submit a job to the server, and block until results are collected
        _pool = self._serve()
        try:
            task = _pool.submit(f, args, globals=globals())
        except pp.DestroyedServerError:
            self._is_alive(None)
        return task()
    pipe.__doc__ = AbstractWorkerPool.pipe.__doc__
    def apipe(self, f, *args, **kwds): # register a callback ?
       #AbstractWorkerPool._AbstractWorkerPool__apipe(self, f, *args, **kwds)
        # submit a job, to be collected later with 'get()'
        _pool = self._serve()
        try:
            task = _pool.submit(f, args, globals=globals())
        except pp.DestroyedServerError:
            self._is_alive(None)
        return ApplyResult(task)
    apipe.__doc__ = AbstractWorkerPool.apipe.__doc__
    ########################################################################
    def __repr__(self):
        mapargs = (self.__class__.__name__, self.ncpus, self.servers)
        return "<pool %s(ncpus=%s, servers=%s)>" % mapargs
    def __get_nodes(self):
        """get the number of nodes used in the map"""
        nodes = self.__nodes
        if nodes == None: nodes = '*'
        return nodes
    def __set_nodes(self, nodes):
        """set the number of nodes used in the map"""
        if nodes is None: nodes = 'autodetect'
        self._serve(nodes=nodes)
        return
    def __get_servers(self):
        """get the servers used in the map"""
        servers = self.__servers
        if servers == (): servers = None
        elif servers == ('*',): servers = '*'
        return servers
    def __set_servers(self, servers):
        """set the servers used in the map"""
        if servers is None: servers = ()
        self._serve(servers=servers)
        #__STATE[self._id].ppservers == [(s.split(':')[0],int(s.split(':')[1])) for s in servers]
        # we could check if the above is true... for now we will just be lazy
        # we could also convert lists to tuples... again, we'll be lazy
        # XXX: throws "socket error" when autodiscovery service is enabled
        return
    ########################################################################
    def restart(self, force=False):
        "restart a closed pool"
        _pool = __STATE.get(self._id, None)
        if self._equals(_pool):
            if not force: self._is_alive(_pool, negate=True)
            # 'clear' and 'serve'
            if self._exiting: # i.e. is destroyed
                self._clear()
            else: # only closed, so just 'reopen'
                _pool._exiting = False 
                #NOTE: setting pool._exiting = False *may* hang python!
            _pool = self._serve()
        return
    def _is_alive(self, server=None, negate=False, run=True):
        RUN,CLOSE,TERMINATE = 0,1,2
        pool = lambda :None
        if server is None:
            pool._state = RUN if negate else CLOSE
        else:
            pool._state = server._exiting
        if negate and run: # throw error if alive (exiting=True)
            assert pool._state != RUN
        elif negate: # throw error if alive (exiting=True)
            assert pool._state in (CLOSE, TERMINATE)
        else:     # throw error if not alive (exiting=False)
            raise ValueError("Pool not running")
    def _equals(self, server):
        "check if the server is compatible"
        if not server:
            return False
        _nodes = cpu_count() if self.__nodes is None else self.__nodes
        if _nodes != server.get_ncpus():
            return False
        _auto = [('*',)] if server.auto_ppservers else []
        _servers = sorted(server.ppservers + _auto)
        _servers = [':'.join((str(i) for i in tup)) for tup in _servers]
        return sorted(self.__servers) == _servers
    def close(self):
        "close the pool to any new jobs"
        _pool = __STATE.get(self._id, None)
        if self._equals(_pool):
            _pool._exiting = True
        return
    def terminate(self):
        "a more abrupt close"
        self.close()
        self.join()
        return
    def join(self):
        "cleanup the closed worker processes"
        _pool = __STATE.get(self._id, None)
        if self._equals(_pool):
            self._is_alive(_pool, negate=True, run=False)
            _pool.destroy()
            self._exiting = True # i.e. is destroyed
        return
    # interface
    ncpus = property(__get_nodes, __set_nodes)
    nodes = property(__get_nodes, __set_nodes)
    servers = property(__get_servers, __set_servers)
    __state__ = __STATE
    pass


# backward compatibility
ParallelPythonPool = ParallelPool

# EOF

