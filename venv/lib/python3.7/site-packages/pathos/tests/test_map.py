#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE

import time

verbose = False
delay = 0.01
items = 100


def busy_add(x,y, delay=0.01):
    import time
    for n in range(x):
       x += n
    for n in range(y):
       y -= n
    time.sleep(delay)
    return x + y


def timed_pool(pool, items=100, delay=0.1, verbose=False):
    _x = range(-items//2,items//2,2)
    _y = range(len(_x))
    _d = [delay]*len(_x)

    if verbose: print(pool)
    start = time.time()
    res = pool.map(busy_add, _x, _y, _d)
    _t = time.time() - start
    if verbose: print("time to queue: %s" % _t)
    start = time.time()
    _sol_ = list(res)
    t_ = time.time() - start
    if verbose: print("time to results: %s\n" %  t_)
    return _sol_


class BuiltinPool(object):
    def map(self, *args):
        return list(map(*args))

std = timed_pool(BuiltinPool(), items, delay=0, verbose=False)


def test_serial():
    from pathos.pools import SerialPool as PS
    pool = PS()
    res = timed_pool(pool, items, delay, verbose)
    assert res == std


def test_pp():
    from pathos.pools import ParallelPool as PPP
    pool = PPP(servers=('localhost:5653','localhost:2414'))
    res = timed_pool(pool, items, delay, verbose)
    assert res == std


def test_processing():
    from pathos.pools import ProcessPool as MPP
    pool = MPP()
    res = timed_pool(pool, items, delay, verbose)
    assert res == std


def test_threading():
    from pathos.pools import ThreadPool as MTP
    pool = MTP()
    res = timed_pool(pool, items, delay, verbose)
    assert res == std


if __name__ == '__main__':
    if verbose:
        print("CONFIG: delay = %s" % delay)
        print("CONFIG: items = %s" % items)
        print("")

    from pathos.helpers import freeze_support, shutdown
    freeze_support()
    test_serial()
    test_pp()
    test_processing()
    test_threading()
    shutdown()
