#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
"""
This module contains functions for profiling in other threads and processes.

Functions for identifying a thread/process:
    process_id   - get the identifier (process id) for the current process
    thread_id    - get the identifier for the current thread

Functions for controlling profiling:
    enable_profiling - initialize a profiler in the current thread/process
    start_profiling  - begin profiling everything in the current thread/process
    stop_profiling   - stop profiling everything in the current thread/process
    disable_profiling - remove the profiler from the current thread/process

Functions that control profile statstics (pstats) output
    clear_stats  - clear stored pstats from the current thread/process
    get_stats    - get stored pstats for the current thread/process
    print_stats  - print stored pstats for the current thread/process
    dump_stats   - dump stored pstats for the current thread/process

Functions that add/remove profiling:
    profiled     - decorator to add profiling to a function
    not_profiled - decorator to remove profiling from a function
    profile     -  decorator for profiling a function (will enable_profiling)

Usage
=====

Typical calls to pathos profiling will roughly follow this example::

    >>> import time
    >>> import random
    >>> import pathos.profile as pr
    >>>
    >>> # build a worker function
    >>> def _work(i):
    ...     x = random.random()
    ...     time.sleep(x)
    ...     return (i,x)
    >>>
    >>> # generate a 'profiled' work function
    >>> config = dict(gen=pr.process_id)
    >>> work = pr.profiled(**config)(_work)
    >>> 
    >>> # enable profiling
    >>> pr.enable_profiling()
    >>> 
    >>> # profile the work (not the map internals) in the main process
    >>> for i in map(work, range(-10,0)):
    ...     print(i)
    ...
    >>> # profile the map in the main process, and work in the other process
    >>> from pathos.helpers import mp
    >>> pool = mp.Pool(10)
    >>> _uimap = pr.profiled(**config)(pool.imap_unordered)
    >>> for i in _uimap(work, range(-10,0)):
    ...     print(i)
    ...
    >>> # deactivate all profiling
    >>> pr.disable_profiling() # in the main process
    >>> tuple(_uimap(pr.disable_profiling, range(10))) # in the workers
    >>> for i in _uimap(work, range(-20,-10)):
    ...     print(i)
    ...
    >>> # re-activate profiling
    >>> pr.enable_profiling()
    >>> 
    >>> # print stats for profile of 'import math' in another process
    >>> def test_import(module):
    ...    __import__(module)
    ...
    >>> import pathos.pools as pp
    >>> pool = pp.ProcessPool(1)
    >>> pr.profile('cumulative', pipe=pool.pipe)(test_import, 'pox')
         10 function calls in 0.003 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.003    0.003 <stdin>:1(test_import)
        1    0.002    0.002    0.003    0.003 {__import__}
        1    0.001    0.001    0.001    0.001 __init__.py:8(<module>)
        1    0.000    0.000    0.000    0.000 shutils.py:11(<module>)
        1    0.000    0.000    0.000    0.000 _disk.py:15(<module>)
        1    0.000    0.000    0.000    0.000 {eval}
        1    0.000    0.000    0.000    0.000 utils.py:11(<module>)
        1    0.000    0.000    0.000    0.000 <string>:1(<module>)
        1    0.000    0.000    0.000    0.000 info.py:2(<module>)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}


    >>> pool.close()
    >>> pool.join()
    >>> pool.clear()


Notes
=====

This module leverages the python's cProfile module, and is primarily a
high-level interface to that module that strives to make profiling in
a different thread or process easier.  The use of pathos.pools are suggested,
however are not required (as seen in the example above).

In many cases, profiling in another thread is not necessary, and either of
the following can be sufficient/better for timing and profiling::

    $ python -c "import time; s=time.time(); import pathos; print (time.time()-s)"
    $ python -c "import cProfile; p=cProfile.Profile(); p.enable(); import pathos; p.print_stats('cumulative')"

This module was inspired by: http://stackoverflow.com/a/32522579/4646678.
"""
# module-level handle to the profiler instance for the current thread/process
profiler = None

def process_id():
    "get the identifier (process id) for the current process"
    from pathos.helpers import mp
    return mp.current_process().pid

def thread_id():
    "get the identifier for the current thread"
    import threading as th
    return th.current_thread().ident

class profiled(object):
    "decorator for profiling a function (does not call enable profiling)"
    def __init__(self, gen=None, prefix='id-', suffix='.prof'):
        """y=gen(), with y an indentifier (e.g. current_process().pid)

Important class members:
    prefix	- string prefix for pstats filename [default: 'id-']
    suffix      - string suffix for pstats filename [default: '.prof']
    pid         - function for obtaining id of current process/thread
    sort        - integer index of column in pstats output for sorting

Example:
    >>> import time
    >>> import random
    >>> import pathos.profile as pr
    >>>
    >>> config = dict(gen=pr.process_id)
    >>> @pr.profiled(**config)
    ... def work(i):
    ...     x = random.random()
    ...     time.sleep(x)
    ...     return (i,x)
    ...
    >>> pr.enable_profiling()
    >>> # profile the work (not the map internals); write to file for pstats
    >>> for i in map(work, range(-10,0)):
    ...     print(i)
    ...

NOTE: If gen is a bool or string, then sort=gen and pid is not used.
      Otherwise, pid=gen and sort is not used. Output can be ordered
      by setting gen to one of the following:
      'calls'      - call count
      'cumulative' - cumulative time
      'cumtime'    - cumulative time
      'file'       - file name
      'filename'   - file name
      'module'     - file name
      'ncalls'     - call count
      'pcalls'     - primitive call count
      'line'       - line number
      'name'       - function name
      'nfl'        - name/file/line
      'stdname'    - standard name
      'time'       - internal time
      'tottime'    - internal time
        """
        self.prefix = prefix
        self.suffix= suffix
        #XXX: tricky: if gen is bool/str then print, else dump with gen=id_gen
        if type(gen) in (bool, str):
            self.sort = -1 if type(gen) is bool else gen
            self.pid = str
        else:
            self.sort = -1
            self.pid = process_id if gen is None else gen
    def __call__(self, f):
        def proactive(*args, **kwds):
            try:
                profiler.enable()
                doit = True
            except AttributeError: doit = False
            except NameError: doit = False
            res = f(*args, **kwds)
            if doit:
                profiler.disable() # XXX: option to not dump?
                if self.pid is str: profiler.print_stats(self.sort)
                else: profiler.dump_stats('%s%s%s' % (self.prefix,self.pid(),self.suffix))
            return res
        proactive.__wrapped__ = f #XXX: conflicts with other __wrapped__ ?
        return proactive

def not_profiled(f):
    "decorator to remove profiling (due to 'profiled') from a function"
    if getattr(f, '__name__', None) == 'proactive':
        _f = getattr(f, '__wrapped__', f)
    else:
        _f = f
    def wrapper(*args, **kwds):
        return _f(*args, **kwds)
    return wrapper

def enable_profiling(*args): #XXX: args ignored (needed for use in map)
    "initialize a profiler instance in the current thread/process"
    global profiler #XXX: better profiler[0] or dict?
    import cProfile
    profiler = cProfile.Profile()  #XXX: access at: pathos.profile.profiler
    return

def start_profiling(*args):
    "begin profiling everything in the current thread/process"
    if profiler is None: enable_profiling()
    try: profiler.enable()
    except AttributeError: pass
    except NameError: pass
    return

def stop_profiling(*args):
    "stop profiling everything in the current thread/process"
    try: profiler.disable()
    except AttributeError: pass
    except NameError: pass
    return

def disable_profiling(*args):
    "remove the profiler instance from the current thread/process"
    global profiler
    if profiler is not None: stop_profiling()
    globals().pop('profiler', None)
    profiler = None
    return

def clear_stats(*args):
    "clear all stored profiling results from the current thread/process"
    try: profiler.clear()
    except AttributeError: pass
    except NameError: pass
    return

def get_stats(*args):
    "get all stored profiling results for the current thread/process"
    try: res = profiler.getstats()
    except AttributeError: pass
    except NameError: pass
    return res

def print_stats(*args, **kwds): #kwds=dict(sort=-1)
    "print all stored profiling results for the current thread/process"
    sort = kwds.get('sort', -1)
    try: profiler.print_stats(sort)
    except AttributeError: pass
    except NameError: pass
    return

def dump_stats(*args, **kwds): # kwds=dict(gen=None, prefix='id-', suffix='.prof'))
    """dump all stored profiling results for the current thread/process

Notes:
    see ``pathos.profile.profiled`` for settings for ``*args`` and ``**kwds``
    """
    config = dict(gen=None, prefix='id-', suffix='.prof')
    config.update(kwds)
    prefix = config['prefix']
    suffix= config['suffix']
    pid = config['gen']
    pid = process_id if pid is None else pid  #XXX: default is str??
    file = '%s%s%s' % (prefix, pid(), suffix)
    try: profiler.dump_stats(file)
    except AttributeError: pass
    except NameError: pass
    return

class profile(object):
    "decorator for profiling a function (will enable profiling)"
    def __init__(self, sort=None, **config):
        """sort is integer index of column in pstats output for sorting

Important class members:
    pipe        - pipe instance in which profiling is active

Example:
    >>> import time
    >>> import random
    >>> import pathos.profile as pr
    >>>
    ... def work():
    ...     x = random.random()
    ...     time.sleep(x)
    ...     return x
    ...
    >>> # profile the work; print pstats info
    >>> pr.profile()(work)
             4 function calls in 0.136 seconds

       Ordered by: standard name

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    0.000    0.000    0.136    0.136 <stdin>:1(work)
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
            1    0.000    0.000    0.000    0.000 {method 'random' of '_random.Random' objects}
            1    0.136    0.136    0.136    0.136 {time.sleep}

    0.1350568110491419
    >>>

NOTE: pipe provided should come from pool built with nodes=1. Other
      configuration keywords (config) are passed to 'pr.profiled'.
      Output can be ordered by setting sort to one of the following:
      'calls'      - call count
      'cumulative' - cumulative time
      'cumtime'    - cumulative time
      'file'       - file name
      'filename'   - file name
      'module'     - file name
      'ncalls'     - call count
      'pcalls'     - primitive call count
      'line'       - line number
      'name'       - function name
      'nfl'        - name/file/line
      'stdname'    - standard name
      'time'       - internal time
      'tottime'    - internal time
        """
        pipe = config.pop('pipe', None)
        if type(sort) not in (bool, type(None)):
            config.update(dict(gen=sort))
        self.config = dict(gen=False) if not bool(config) else config
        from pathos.pools import SerialPool
        if pipe is None:
            self._pool = SerialPool()
            self.pipe = self._pool.pipe
        else:
            self.pipe = pipe
            self._pool = getattr(pipe, '__self__', SerialPool())
        if self._pool.nodes != 1:
            raise ValueError('pipe must draw from a pool with only one node')
        return
    def __call__(self, function, *args, **kwds):
       #self._pool.nodes, nodes = 1, self._pool.nodes #XXX: skip this?
        self.pipe(enable_profiling, None)
        result = self.pipe(profiled(**self.config)(function), *args, **kwds)
        self.pipe(disable_profiling, None)
       #self._pool.nodes = nodes #XXX: skip this?
        return result

"""
def _enable_profiling(f): #FIXME: gradual: only applied to *new* workers
    "activate profiling for the given function in the current thread"
    def func(*arg, **kwd):
        enable_profiling()
        #XXX: include f under profiler or above?
        return f(*arg, **kwd)
#   func.__wrapped__ = f #XXX: conflict with other usings __wrapped__ 
    return func

def _disable_profiling(f): #FIXME: gradual: only applied to *new* workers
    "deactivate profiling for the given function in the current thread"
    try: _f = f.__wrapped__
    except AttributeError: _f = f
    def func(*arg, **kwd):
        disable_profiling()
        #XXX: include f under profiler or above?
        return _f(*arg, **kwd)
    func.__wrapped__ = _f
    return func

def profiling(pool):
    "decorator for initializing profiling functions called within a pool"
    def wrapper(*args, **kwds):
        initializer = kwds.get('initializer', None)
        pool._rinitializer = initializer
        if initializer is None: initializer = lambda *x,**y: (x,y)
        kwds['initializer'] = _enable_profiling(initializer)
        return pool(*args, **kwds)

    return wrapper
"""


# EOF
