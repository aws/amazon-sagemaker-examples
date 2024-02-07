#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
"""
minimal interface to python's multiprocessing module

Notes:
    This module has been deprecated in favor of ``pathos.pools``.
"""

from pathos.multiprocessing import ProcessPool, __STATE
from pathos.threading import ThreadPool #XXX: thread __STATE not imported
from pathos.helpers import cpu_count
mp = ProcessPool() #FIXME: don't do this
tp = ThreadPool() #FIXME: don't do this

__all__ = ['mp_map']

# backward compatibility
#FIXME: deprecated... and buggy!  (fails to dill on imap/uimap)
def mp_map(function, sequence, *args, **kwds):
    '''extend python's parallel map function to multiprocessing

Args:
    function - target function
    sequence - sequence to process in parallel
    nproc - number of 'local' cpus to use  [defaut = 'autodetect']
    type - processing type ['blocking', 'non-blocking', 'unordered']
    threads - if True, use threading instead of multiprocessing
    '''
    processes = cpu_count()
    proctype = 'blocking'
    threads = False
    if 'nproc' in kwds:
        processes = kwds['nproc']
        kwds.pop('nproc')
        # provide a default that is not a function call
        if processes == None: processes = cpu_count()
    if 'type' in kwds:
        proctype = kwds['type']
        kwds.pop('type')
    if 'threads' in kwds:
        threads = kwds['threads']
        kwds.pop('threads')
    # remove all the junk kwds that are added due to poor design!
    if 'nnodes' in kwds: kwds.pop('nnodes')
    if 'nodes' in kwds: kwds.pop('nodes')
    if 'launcher' in kwds: kwds.pop('launcher')
    if 'mapper' in kwds: kwds.pop('mapper')
    if 'queue' in kwds: kwds.pop('queue')
    if 'timelimit' in kwds: kwds.pop('timelimit')
    if 'scheduler' in kwds: kwds.pop('scheduler')
    if 'ncpus' in kwds: kwds.pop('ncpus')
    if 'servers' in kwds: kwds.pop('servers')

    if proctype in ['blocking']:
        if not threads:
            return mp.map(function,sequence,*args,**kwds)
        else:
            return tp.map(function,sequence,*args,**kwds)
    elif proctype in ['unordered']:
        if not threads:
            return mp.uimap(function,sequence,*args,**kwds)
        else:
            return tp.uimap(function,sequence,*args,**kwds)
    elif proctype in ['non-blocking', 'ordered']:
        if not threads:
            return mp.imap(function,sequence,*args,**kwds)
        else:
            return tp.imap(function,sequence,*args,**kwds)
    # default
    if not threads:
        return mp.map(function,sequence,*args,**kwds)
    else:
        return tp.map(function,sequence,*args,**kwds)



if __name__ == '__main__':
  pass

