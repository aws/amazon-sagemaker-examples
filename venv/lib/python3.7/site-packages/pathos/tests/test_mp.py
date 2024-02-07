#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
import time

def test_mp():
    # instantiate and configure the worker pool
    from pathos.pools import ProcessPool
    pool = ProcessPool(nodes=4)

    _result = list(map(pow, [1,2,3,4], [5,6,7,8]))

    # do a blocking map on the chosen function
    result = pool.map(pow, [1,2,3,4], [5,6,7,8])
    assert result == _result

    # do a non-blocking map, then extract the result from the iterator
    result_iter = pool.imap(pow, [1,2,3,4], [5,6,7,8])
    result = list(result_iter)
    assert result == _result

    # do an asynchronous map, then get the results
    result_queue = pool.amap(pow, [1,2,3,4], [5,6,7,8])
    result = result_queue.get()
    assert result == _result

    # test ProcessPool keyword argument propagation
    pool.clear()
    pool = ProcessPool(nodes=4, initializer=lambda: time.sleep(0.6))
    start = time.monotonic()
    result = pool.map(pow, [1,2,3,4], [5,6,7,8])
    end = time.monotonic()
    assert result == _result
    assert end - start > 0.5

def test_tp():
    # instantiate and configure the worker pool
    from pathos.pools import ThreadPool
    pool = ThreadPool(nodes=4)

    _result = list(map(pow, [1,2,3,4], [5,6,7,8]))

    # do a blocking map on the chosen function
    result = pool.map(pow, [1,2,3,4], [5,6,7,8])
    assert result == _result

    # do a non-blocking map, then extract the result from the iterator
    result_iter = pool.imap(pow, [1,2,3,4], [5,6,7,8])
    result = list(result_iter)
    assert result == _result

    # do an asynchronous map, then get the results
    result_queue = pool.amap(pow, [1,2,3,4], [5,6,7,8])
    result = result_queue.get()
    assert result == _result

    # test ThreadPool keyword argument propagation
    pool.clear()
    pool = ThreadPool(nodes=4, initializer=lambda: time.sleep(0.6))
    start = time.monotonic()
    result = pool.map(pow, [1,2,3,4], [5,6,7,8])
    end = time.monotonic()
    assert result == _result
    assert end - start > 0.5


def test_chunksize():
    # instantiate and configure the worker pool
    from pathos.pools import ProcessPool, _ProcessPool, ThreadPool
    from pathos.helpers.mp_helper import starargs as star
    pool = _ProcessPool(4)
    ppool = ProcessPool(4)
    tpool = ThreadPool(4)

    # do a blocking map on the chosen function
    result1 = pool.map(star(pow), zip([1,2,3,4],[5,6,7,8]), 1)
    assert result1 == ppool.map(pow, [1,2,3,4], [5,6,7,8], chunksize=1)
    assert result1 == tpool.map(pow, [1,2,3,4], [5,6,7,8], chunksize=1)
    result0 = pool.map(star(pow), zip([1,2,3,4],[5,6,7,8]), 0)
    assert result0 == ppool.map(pow, [1,2,3,4], [5,6,7,8], chunksize=0)
    assert result0 == tpool.map(pow, [1,2,3,4], [5,6,7,8], chunksize=0)

    # do an asynchronous map, then get the results
    result1 = pool.map_async(star(pow), zip([1,2,3,4],[5,6,7,8]), 1).get()
    assert result1 == ppool.amap(pow, [1,2,3,4], [5,6,7,8], chunksize=1).get()
    assert result1 == tpool.amap(pow, [1,2,3,4], [5,6,7,8], chunksize=1).get()
    result0 = pool.map_async(star(pow), zip([1,2,3,4],[5,6,7,8]), 0).get()
    assert result0 == ppool.amap(pow, [1,2,3,4], [5,6,7,8], chunksize=0).get()
    assert result0 == tpool.amap(pow, [1,2,3,4], [5,6,7,8], chunksize=0).get()

    # do a non-blocking map, then extract the result from the iterator
    result1 = list(pool.imap(star(pow), zip([1,2,3,4],[5,6,7,8]), 1))
    assert result1 == list(ppool.imap(pow, [1,2,3,4], [5,6,7,8], chunksize=1))
    assert result1 == list(tpool.imap(pow, [1,2,3,4], [5,6,7,8], chunksize=1))
    try:
        list(pool.imap(star(pow), zip([1,2,3,4],[5,6,7,8]), 0))
        error = AssertionError
    except Exception:
        import sys
        error = sys.exc_info()[0]
    try:
        list(ppool.imap(pow, [1,2,3,4], [5,6,7,8], chunksize=0))
        assert False
    except error:
        pass
    except Exception:
        import sys
        e = sys.exc_info()[1]
        raise AssertionError(str(e))
    try:
        list(tpool.imap(pow, [1,2,3,4], [5,6,7,8], chunksize=0))
        assert False
    except error:
        pass
    except Exception:
        import sys
        e = sys.exc_info()[1]
        raise AssertionError(str(e))

    # do a non-blocking map, then extract the result from the iterator
    res1 = sorted(pool.imap_unordered(star(pow), zip([1,2,3,4],[5,6,7,8]), 1))
    assert res1 == sorted(ppool.uimap(pow, [1,2,3,4], [5,6,7,8], chunksize=1))
    assert res1 == sorted(tpool.uimap(pow, [1,2,3,4], [5,6,7,8], chunksize=1))
    try:
        sorted(pool.imap_unordered(star(pow), zip([1,2,3,4],[5,6,7,8]), 0))
        error = AssertionError
    except Exception:
        import sys
        error = sys.exc_info()[0]
    try:
        sorted(ppool.uimap(pow, [1,2,3,4], [5,6,7,8], chunksize=0))
        assert False
    except error:
        pass
    except Exception:
        import sys
        e = sys.exc_info()[1]
        raise AssertionError(str(e))
    try:
        sorted(tpool.uimap(pow, [1,2,3,4], [5,6,7,8], chunksize=0))
        assert False
    except error:
        pass
    except Exception:
        import sys
        e = sys.exc_info()[1]
        raise AssertionError(str(e))


if __name__ == '__main__':
    from pathos.helpers import freeze_support, shutdown
    freeze_support()
    test_mp()
    test_tp()
    test_chunksize()
    shutdown()
