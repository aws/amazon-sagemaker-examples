#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2022-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
"""
maps: stand-alone map-like objects using lazy pool instantiation
"""

class Map(object):

    def __init__(self, pool=None, *args, **kwds):
        """map instance with internal lazy pool instantiation

    Args:
        pool: pool object (i.e. pathos.pools.ProcessPool)
        *args: positional arguments for pool initialization
        **kwds: keyword arguments for pool initialization
        close: if True, close the pool to any new jobs [Default: False] 
        join: if True, reclaim the pool's closed workers [Default: False]
        clear: if True, delete the pool singleton [Default: False]

    NOTE: if a pool object is not provided, a builtins.map will be
        used with the returned iterator cast to a list.
    NOTE: pools from both multiprocess and pathos.pools can be used,
        however the behavior is slightly different. Pools from both
        pathos and multiprocess have close and join methods, to close
        the pool to new jobs, and to shut down the pool's workers.
        Pools from pathos, however, are launched as singletons, so
        they also include a clear method that deletes the singleton.
        In either case, a pool that has been "closed" will throw a
        ValueError if map is then called, and similarly, a ValueError
        will be thrown if join is called before a pool is "closed".
        The major difference is that if a pathos.pool is closed, the
        map instance cannot run new jobs until "clear" is called,
        while a new multiprocess pool will be created each time the
        map is executed. This leads to pathos.pools generally being
        called with either ``clear=True`` or ``clear=False``, and pools
        from multprocess either using ``close=True`` or ``join=True`` or
        both. Some hierarchical parallel workflows are not allowed,
        and will result in an error being thrown; however, changing
        close, join, or clear can often remove the error.
        """
        self._clear = kwds.pop('clear', False)
        self._join = kwds.pop('join', self._clear)
        self._close = kwds.pop('close', self._join)
        self.pool = None if pool is map else pool
        self.args = args
        self.kwds = kwds
        self._pool = None

    def __call__(self, func, *args, **kwds):
        """instantiate a pool and execute the pool's map

    Args:
        func: function object to map
        *args: positional arguments for map
        **kwds: keyword arguments for map

    Returns:
        results from execution of ``map(func, *args, **kwds)``

    NOTE: initializes a new worker pool with each call
        """
        if self.pool is None: #XXX: args, kwds?
            return list(map(func, *args)) #XXX: iterator or list?
        self._pool = pool = self.pool(*self.args, **self.kwds)
        result = pool.map(func, *args, **kwds)
        if self._close: self.close()
        if self._join: self.join()
        if self._clear: self.clear()
        return result #NOTE: ValueError on non-running pool

    # function interface
    def __cls__(self):
        return self
    def __meth__(self):
        return self.__call__.__func__
    def __attr__(self):
        return self.__call__.__get__
    __self__ = property(__cls__)
    __func__ = property(__meth__)
    __get__ = property(__attr__)

    def close(self):
        "close the map to any new jobs"
        try:
            self._pool.close()
        except AttributeError:
            pass

    def join(self): #NOTE: ValueError on running pool
        "reclaim the closed workers"
        try:
            self._pool.join()
        except AttributeError:
            pass

    def clear(self):
        "remove pool singleton, if exists"
        try:
            self._pool.clear()
        except AttributeError:
            pass

    def __del__(self):
        """shutdown the worker pool and tidy up
        """
        try:
            self.close()
            self.join()
            self.clear() #XXX: clear or not?
        except Exception:
            pass
        self._pool = None


class Smap(Map):

    def __init__(self, pool=None, *args, **kwds):
        """starmap instance with internal lazy pool instantiation

    Args:
        pool: pool object (i.e. pathos.pools.ProcessPool)
        *args: positional arguments for pool initialization
        **kwds: keyword arguments for pool initialization
        close: if True, close the pool to any new jobs [Default: False] 
        join: if True, reclaim the pool's closed workers [Default: False]
        clear: if True, delete the pool singleton [Default: False]

    NOTE: if a pool object is not provided, an itertools.starmap will
        be used with the returned iterator cast to a list.
    NOTE: pools from both multiprocess and pathos.pools can be used,
        however the behavior is slightly different. Pools from both
        pathos and multiprocess have close and join methods, to close
        the pool to new jobs, and to shut down the pool's workers.
        Pools from pathos, however, are launched as singletons, so
        they also include a clear method that deletes the singleton.
        In either case, a pool that has been "closed" will throw a
        ValueError if map is then called, and similarly, a ValueError
        will be thrown if join is called before a pool is "closed".
        The major difference is that if a pathos.pool is closed, the
        map instance cannot run new jobs until "clear" is called,
        while a new multiprocess pool will be created each time the
        map is executed. This leads to pathos.pools generally being
        called with either ``clear=True`` or ``clear=False``, and pools
        from multprocess either using ``close=True`` or ``join=True`` or
        both. Some hierarchical parallel workflows are not allowed,
        and will result in an error being thrown; however, changing
        close, join, or clear can often remove the error.
        """
        super().__init__(pool, *args, **kwds)

    def __call__(self, func, *args, **kwds):
        """instantiate a pool and execute the pool's starmap

    Args:
        func: function object to map
        *args: positional arguments for starmap
        **kwds: keyword arguments for starmap

    Returns:
        results from execution of ``starmap(func, *args, **kwds)``

    NOTE: initializes a new worker pool with each call
        """
        if self.pool is None: #XXX: args, kwds?
            from itertools import starmap
            return list(starmap(func, *args)) #XXX: iterator or list?
        self._pool = pool = self.pool(*self.args, **self.kwds)
        smap = getattr(pool, 'smap', getattr(pool, 'starmap', None))
        if smap is None:
            result = pool.map(lambda x: func(*x), *args, **kwds)
        else:
            result = smap(func, *args, **kwds)
        if self._close: self.close()
        if self._join: self.join()
        if self._clear: self.clear()
        return result


class Imap(Map):

    def __init__(self, pool=None, *args, **kwds):
        """map iterator with internal lazy pool instantiation

    Args:
        pool: pool object (i.e. pathos.pools.ProcessPool)
        *args: positional arguments for pool initialization
        **kwds: keyword arguments for pool initialization
        close: if True, close the pool to any new jobs [Default: False] 
        join: if True, reclaim the pool's closed workers [Default: False]
        clear: if True, delete the pool singleton [Default: False]

    NOTE: if a pool object is not provided, a builtins.map will be
        used.
    NOTE: pools from both multiprocess and pathos.pools can be used,
        however the behavior is slightly different. Pools from both
        pathos and multiprocess have close and join methods, to close
        the pool to new jobs, and to shut down the pool's workers.
        Pools from pathos, however, are launched as singletons, so
        they also include a clear method that deletes the singleton.
        In either case, a pool that has been "closed" will throw a
        ValueError if map is then called, and similarly, a ValueError
        will be thrown if join is called before a pool is "closed".
        The major difference is that if a pathos.pool is closed, the
        map instance cannot run new jobs until "clear" is called,
        while a new multiprocess pool will be created each time the
        map is executed. This leads to pathos.pools generally being
        called with either ``clear=True`` or ``clear=False``, and pools
        from multprocess either using ``close=True`` or ``join=True`` or
        both. Some hierarchical parallel workflows are not allowed,
        and will result in an error being thrown; however, changing
        close, join, or clear can often remove the error.
        """
        super().__init__(pool, *args, **kwds)

    def __call__(self, func, *args, **kwds):
        """instantiate a pool and execute the pool's map iterator

    Args:
        func: function object to map
        *args: positional arguments for map iterator
        **kwds: keyword arguments for map iterator

    Returns:
        results from execution of ``map(func, *args, **kwds)`` iterator

    NOTE: initializes a new worker pool with each call
        """
        if self.pool is None: #XXX: args, kwds?
            return map(func, *args)
        self._pool = pool = self.pool(*self.args, **self.kwds)
        imap = getattr(pool, 'imap', None)
        if imap is None: #NOTE: should not happen
            return NotImplemented
        result = imap(func, *args, **kwds)
        if self._close: self.close()
        if self._join: self.join()
        if self._clear: self.clear()
        return result


class Amap(Map):

    def __init__(self, pool=None, *args, **kwds):
        """async map instance with internal lazy pool instantiation

    Args:
        pool: pool object (i.e. pathos.pools.ProcessPool)
        *args: positional arguments for pool initialization
        **kwds: keyword arguments for pool initialization
        close: if True, close the pool to any new jobs [Default: False] 
        join: if True, reclaim the pool's closed workers [Default: False]
        clear: if True, delete the pool singleton [Default: False]

    NOTE: if a pool object is not provided, NotImplemented is returned
        upon use.
    NOTE: pools from both multiprocess and pathos.pools can be used,
        however the behavior is slightly different. Pools from both
        pathos and multiprocess have close and join methods, to close
        the pool to new jobs, and to shut down the pool's workers.
        Pools from pathos, however, are launched as singletons, so
        they also include a clear method that deletes the singleton.
        In either case, a pool that has been "closed" will throw a
        ValueError if map is then called, and similarly, a ValueError
        will be thrown if join is called before a pool is "closed".
        The major difference is that if a pathos.pool is closed, the
        map instance cannot run new jobs until "clear" is called,
        while a new multiprocess pool will be created each time the
        map is executed. This leads to pathos.pools generally being
        called with either ``clear=True`` or ``clear=False``, and pools
        from multprocess either using ``close=True`` or ``join=True`` or
        both. Some hierarchical parallel workflows are not allowed,
        and will result in an error being thrown; however, changing
        close, join, or clear can often remove the error.
        """
        super().__init__(pool, *args, **kwds)

    def __call__(self, func, *args, **kwds):
        """instantiate a pool and execute the pool's async map

    Args:
        func: function object to map
        *args: positional arguments for async map
        **kwds: keyword arguments for async map

    Returns:
        results from execution of async ``map(func, *args, **kwds)``

    NOTE: initializes a new worker pool with each call
        """
        if self.pool is None: #XXX: args, kwds?
            return NotImplemented
        self._pool = pool = self.pool(*self.args, **self.kwds)
        amap = getattr(pool, 'amap', getattr(pool, 'map_async', None))
        if amap is None: #NOTE: should not happen
            return NotImplemented
        result = amap(func, *args, **kwds)
        if self._close: self.close()
        if self._join: self.join()
        if self._clear: self.clear()
        return result


class Asmap(Map):

    def __init__(self, pool=None, *args, **kwds):
        """async starmap instance with internal lazy pool instantiation

    Args:
        pool: pool object (i.e. pathos.pools.ProcessPool)
        *args: positional arguments for pool initialization
        **kwds: keyword arguments for pool initialization
        close: if True, close the pool to any new jobs [Default: False] 
        join: if True, reclaim the pool's closed workers [Default: False]
        clear: if True, delete the pool singleton [Default: False]

    NOTE: if a pool object is not provided, NotImplemented is returned
        upon use.
    NOTE: pools from both multiprocess and pathos.pools can be used,
        however the behavior is slightly different. Pools from both
        pathos and multiprocess have close and join methods, to close
        the pool to new jobs, and to shut down the pool's workers.
        Pools from pathos, however, are launched as singletons, so
        they also include a clear method that deletes the singleton.
        In either case, a pool that has been "closed" will throw a
        ValueError if map is then called, and similarly, a ValueError
        will be thrown if join is called before a pool is "closed".
        The major difference is that if a pathos.pool is closed, the
        map instance cannot run new jobs until "clear" is called,
        while a new multiprocess pool will be created each time the
        map is executed. This leads to pathos.pools generally being
        called with either ``clear=True`` or ``clear=False``, and pools
        from multprocess either using ``close=True`` or ``join=True`` or
        both. Some hierarchical parallel workflows are not allowed,
        and will result in an error being thrown; however, changing
        close, join, or clear can often remove the error.
        """
        super().__init__(pool, *args, **kwds)

    def __call__(self, func, *args, **kwds):
        """instantiate a pool and execute the pool's async starmap

    Args:
        func: function object to map
        *args: positional arguments for async starmap
        **kwds: keyword arguments for async starmap

    Returns:
        results from execution of async ``starmap(func, *args, **kwds)``

    NOTE: initializes a new worker pool with each call
        """
        if self.pool is None: #XXX: args, kwds?
            return NotImplemented
        self._pool = pool = self.pool(*self.args, **self.kwds)
        asmap = getattr(pool, 'asmap', getattr(pool, 'starmap_async', None))
        if asmap is None:
            result = pool.amap(lambda x: func(*x), *args, **kwds)
        else:
            result = asmap(func, *args, **kwds)
        if self._close: self.close()
        if self._join: self.join()
        if self._clear: self.clear()
        return result


class Uimap(Map):

    def __init__(self, pool=None, *args, **kwds):
        """unordered map iterator with internal lazy pool instantiation

    Args:
        pool: pool object (i.e. pathos.pools.ProcessPool)
        *args: positional arguments for pool initialization
        **kwds: keyword arguments for pool initialization
        close: if True, close the pool to any new jobs [Default: False] 
        join: if True, reclaim the pool's closed workers [Default: False]
        clear: if True, delete the pool singleton [Default: False]

    NOTE: if a pool object is not provided, NotImplemented is returned
        upon use.
    NOTE: pools from both multiprocess and pathos.pools can be used,
        however the behavior is slightly different. Pools from both
        pathos and multiprocess have close and join methods, to close
        the pool to new jobs, and to shut down the pool's workers.
        Pools from pathos, however, are launched as singletons, so
        they also include a clear method that deletes the singleton.
        In either case, a pool that has been "closed" will throw a
        ValueError if map is then called, and similarly, a ValueError
        will be thrown if join is called before a pool is "closed".
        The major difference is that if a pathos.pool is closed, the
        map instance cannot run new jobs until "clear" is called,
        while a new multiprocess pool will be created each time the
        map is executed. This leads to pathos.pools generally being
        called with either ``clear=True`` or ``clear=False``, and pools
        from multprocess either using ``close=True`` or ``join=True`` or
        both. Some hierarchical parallel workflows are not allowed,
        and will result in an error being thrown; however, changing
        close, join, or clear can often remove the error.
        """
        super().__init__(pool, *args, **kwds)

    def __call__(self, func, *args, **kwds):
        """instantiate a pool and execute the pool's unordered map iterator

    Args:
        func: function object to map
        *args: positional arguments for unordered map iterator
        **kwds: keyword arguments for unordered map iterator

    Returns:
        results from execution of unordered ``map(func, *args, **kwds)`` iterator

    NOTE: initializes a new worker pool with each call
        """
        if self.pool is None: #XXX: args, kwds?
            return NotImplemented
        self._pool = pool = self.pool(*self.args, **self.kwds)
        uimap = getattr(pool, 'uimap', getattr(pool, 'imap_unordered', None))
        if uimap is None: #NOTE: should not happen
            return NotImplemented
        result = uimap(func, *args, **kwds)
        if self._close: self.close()
        if self._join: self.join()
        if self._clear: self.clear()
        return result


class Ismap(Map):

    def __init__(self, pool=None, *args, **kwds):
        """starmap iterator with internal lazy pool instantiation

    Args:
        pool: pool object (i.e. pathos.pools.ProcessPool)
        *args: positional arguments for pool initialization
        **kwds: keyword arguments for pool initialization
        close: if True, close the pool to any new jobs [Default: False] 
        join: if True, reclaim the pool's closed workers [Default: False]
        clear: if True, delete the pool singleton [Default: False]

    NOTE: if a pool object is not provided, an itertools.starmap will
        be used.
    NOTE: pools from both multiprocess and pathos.pools can be used,
        however the behavior is slightly different. Pools from both
        pathos and multiprocess have close and join methods, to close
        the pool to new jobs, and to shut down the pool's workers.
        Pools from pathos, however, are launched as singletons, so
        they also include a clear method that deletes the singleton.
        In either case, a pool that has been "closed" will throw a
        ValueError if map is then called, and similarly, a ValueError
        will be thrown if join is called before a pool is "closed".
        The major difference is that if a pathos.pool is closed, the
        map instance cannot run new jobs until "clear" is called,
        while a new multiprocess pool will be created each time the
        map is executed. This leads to pathos.pools generally being
        called with either ``clear=True`` or ``clear=False``, and pools
        from multprocess either using ``close=True`` or ``join=True`` or
        both. Some hierarchical parallel workflows are not allowed,
        and will result in an error being thrown; however, changing
        close, join, or clear can often remove the error.
        """
        super().__init__(pool, *args, **kwds)

    def __call__(self, func, *args, **kwds):
        """instantiate a pool and execute the pool's starmap iterator

    Args:
        func: function object to map
        *args: positional arguments for starmap iterator
        **kwds: keyword arguments for starmap iterator

    Returns:
        results from execution of ``starmap(func, *args, **kwds)`` iterator

    NOTE: initializes a new worker pool with each call
        """
        if self.pool is None: #XXX: args, kwds?
            from itertools import starmap
            return starmap(func, *args)
        self._pool = pool = self.pool(*self.args, **self.kwds)
        ismap = getattr(pool, 'ismap', None)
        if ismap is None:
            result = pool.imap(lambda x: func(*x), *args, **kwds)
        else:
            result = ismap(func, *args, **kwds)
        if self._close: self.close()
        if self._join: self.join()
        if self._clear: self.clear()
        return result


