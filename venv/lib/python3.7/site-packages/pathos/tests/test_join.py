#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2015-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE

from pathos.parallel import *
from pathos.multiprocessing import *
from pathos.threading import *

from pathos.helpers import cpu_count

import sys
import dill
PYPY38 = (sys.hexversion >= 0x30800f0) and dill._dill.IS_PYPY
PoolClosedError = ValueError
PoolRunningError = ValueError

def squared(x):
    return x**2

def check_basic(pool):
    state = pool.__state__
    res = pool.map(squared, range(2))
    assert res == [0, 1]
    res = pool.map(squared, range(2))
    assert res == [0, 1]

    # join needs to be called after close
    try:
        pool.join()
    except PoolRunningError:
        pass
    else:
        raise AssertionError
    pool.close()

    # map fails when closed
    try:
        pool.map(squared, range(2))
    except PoolClosedError:
        pass
    else:
        raise AssertionError

    obj = pool._serve()
    assert obj in list(state.values())

    # serve has no effect on closed
    try:
        pool.map(squared, range(2))
    except PoolClosedError:
        pass
    else:
        raise AssertionError

    # once restarted, map works
    pool.restart()
    res = pool.map(squared, range(2))
    assert res == [0, 1]

    # assorted kicking of the tires...
    pool.close()
    pool.restart()
    res = pool.map(squared, range(2))
    assert res == [0, 1]
    pool.close()
    pool.join()
    pool._serve()
    try:
        pool.map(squared, range(2))
    except PoolClosedError:
        pass
    else:
        raise AssertionError

    pool.join()
    try:
        pool.map(squared, range(2))
    except PoolClosedError:
        pass
    else:
        raise AssertionError

    pool.clear()
    res = pool.map(squared, range(2))
    assert res == [0, 1]

    try:
        pool.join()
    except PoolRunningError:
        pass
    else:
        raise AssertionError

    pool.close()
    pool.join()
    try:
        pool.map(squared, range(2))
    except PoolClosedError:
        pass
    else:
        raise AssertionError
    pool.restart()
    res = pool.map(squared, range(2))
    assert res == [0, 1]
    pool.close()
    pool.join()
    try:
        pool.map(squared, range(2))
    except PoolClosedError:
        pass
    else:
        raise AssertionError

    pool._serve()
    try:
        pool.map(squared, range(2))
    except PoolClosedError:
        pass
    else:
        raise AssertionError

    pool.restart()
    res = pool.map(squared, range(2))
    assert res == [0, 1]
    pool.close()
    try:
        pool.map(squared, range(2))
    except PoolClosedError:
        pass
    else:
        raise AssertionError

    pool.restart()
    res = pool.map(squared, range(2))
    assert res == [0, 1]
    obj = pool._serve()
    assert obj in list(state.values())
    assert len(state) == 1
    pool.terminate()
    pool.clear()
    assert len(state) == 0
    return


def check_nodes(pool):
    state = pool.__state__
    tag = 'fixed' if pool._id == 'fixed' else None
    new_pool = type(pool)

    nodes = cpu_count()
    if nodes < 2: return
    half = nodes//2

    res = pool.map(squared, range(2))
    assert res == [0, 1]
    pool.close()

    # doesn't create a new pool... IS IT BETTER IF IT DOES?
    pool = new_pool(id=tag)
    try:
        pool.map(squared, range(2))
    except PoolClosedError:
        pass
    else:
        raise AssertionError
    
    # creates a new pool (nodes are different)
    def nnodes(pool):
        return getattr(pool, '_'+new_pool.__name__+'__nodes')
    old_nodes = nnodes(pool)
    pool = new_pool(nodes=half, id=tag)
    new_nodes = nnodes(pool)
    if isinstance(pool, ParallelPool):
        print('SKIPPING: new_pool check for ParallelPool')#FIXME
    else:
        res = pool.map(squared, range(2))
        assert res == [0, 1]
        assert new_nodes < old_nodes

    pool.close()
    try:
        pool.map(squared, range(2))
    except PoolClosedError:
        pass
    else:
        raise AssertionError

    # return to old number of nodes
    if tag is None:
        pool.clear() # clear 'half' pool
        pool = new_pool(id=tag)
        pool.restart() # restart old pool
    else: # creates a new pool (update nodes)
        pool = new_pool(id=tag)
    if isinstance(pool, ParallelPool):
        print('SKIPPING: new_pool check for ParallelPool')#FIXME
    else:
        res = pool.map(squared, range(2))
        assert res == [0, 1]
    pool.close()
    # doesn't create a new pool... IS IT BETTER IF IT DOES?
    pool = new_pool(id=tag)
    try:
        pool.map(squared, range(2))
    except PoolClosedError:
        pass
    else:
        raise AssertionError

    assert len(state) == 1
    pool.clear()
    assert len(state) == 0
    pool = new_pool(id=tag)
    res = pool.map(squared, range(2))
    assert res == [0, 1]
    assert len(state) == 1
    pool.terminate()
    assert len(state) == 1
    pool.clear()
    assert len(state) == 0
    return


def check_rename(pool):
    state = pool.__state__
    new_pool = type(pool)
    res = pool.map(squared, range(2))
    assert res == [0, 1]
    old_id = pool._id

    # change the 'id'
    pool._id = 'foobar'
    pool = new_pool() # blow away the 'id' change
    res = pool.map(squared, range(2))
    assert res == [0, 1]
    assert len(state) == 1
    assert 'foobar' not in list(state.keys())

    # change the 'id', but don't re-init
    pool._id = 'foobar'
    res = pool.map(squared, range(2))
    assert res == [0, 1]
    assert len(state) == 2
    assert 'foobar' in list(state.keys())

    pool.close()     
    try:
        pool.map(squared, range(2))
    except PoolClosedError:
        pass
    else:
        raise AssertionError
    pool.terminate()
    assert len(state) == 2
    assert 'foobar' in list(state.keys())
    pool.clear()
    assert len(state) == 1
    assert 'foobar' not in list(state.keys())

    pool._id = old_id
    res = pool.map(squared, range(2))
    assert res == [0, 1]
    pool.terminate()
    pool.clear()
    assert len(state) == 0
    return

def test_basic():
    check_basic(ThreadPool())
#   check_basic(ProcessPool())
#   check_basic(ParallelPool())

def test_rename():
    check_rename(ThreadPool())
    check_rename(ProcessPool())
    check_rename(ParallelPool())

def test_fixed():
    check_nodes(ThreadPool(id='fixed'))
    check_nodes(ProcessPool(id='fixed'))
    check_nodes(ParallelPool(id='fixed'))

def test_nodes():
    check_nodes(ThreadPool())
    if not PYPY38: #FIXME: fails with "OSError: [Errno 24] Too many open files"
        check_nodes(ProcessPool())
    check_nodes(ParallelPool())


if __name__ == '__main__':
    from pathos.helpers import freeze_support, shutdown
    freeze_support()
    test_basic()
    test_rename()
    test_fixed()
    test_nodes()
    shutdown()
