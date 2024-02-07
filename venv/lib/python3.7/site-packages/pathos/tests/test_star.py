#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE

import time
x = list(range(18))
delay = 0.01
items = 20
maxtries = 20


def busy_add(x,y, delay=0.01):
    import time
    for n in range(x):
       x += n
    for n in range(y):
       y -= n
    time.sleep(delay)
    return x + y

def busy_squared(x):
    import time, random
    time.sleep(0.01*random.random())
    return x*x

def squared(x):
    return x*x

def quad_factory(a=1, b=1, c=0):
    def quad(x):
        return a*x**2 + b*x + c
    return quad

square_plus_one = quad_factory(2,0,1)

x2 = list(map(squared, x))


def check_sanity(pool, verbose=False):
    if verbose:
        print(pool)
        print("x: %s\n" % str(x))

        print(pool.map.__name__)
    # blocking map
    start = time.time()
    res = pool.map(squared, x)
    end = time.time() - start
    assert res == x2
    if verbose:
        print("time to results: %s" % end)
        print("y: %s\n" % str(res))

        print(pool.imap.__name__)
    # iterative map
    start = time.time()
    res = pool.imap(squared, x)
    fin = time.time() - start
    # get result from iterator
    start = time.time()
    res = list(res)
    end = time.time() - start
    assert res == x2
    if verbose:
        print("time to queue: %s" % fin)
        print("time to results: %s" % end)
        print("y: %s\n" % str(res))

        print(pool.amap.__name__)
    # asyncronous map
    start = time.time()
    res = pool.amap(squared, x)
    fin = time.time() - start
    # get result from result object
    start = time.time()
    res = res.get()
    end = time.time() - start
    assert res == x2
    if verbose:
        print("time to queue: %s" % fin)
        print("time to results: %s" % end)
        print("y: %s\n" % str(res))


def check_maps(pool, items=4, delay=0):
    _x = range(-items//2,items//2,2)
    _y = range(len(_x))
    _d = [delay]*len(_x)
    _z = [0]*len(_x)

   #print(map)
    res1 = list(map(squared, _x))
    res2 = list(map(busy_add, _x, _y, _z))

   #print(pool.map)
    _res1 = pool.map(squared, _x)
    _res2 = pool.map(busy_add, _x, _y, _d)
    assert _res1 == res1
    assert _res2 == res2

   #print(pool.imap)
    _res1 = pool.imap(squared, _x)
    _res2 = pool.imap(busy_add, _x, _y, _d)
    assert list(_res1) == res1
    assert list(_res2) == res2

   #print(pool.uimap)
    _res1 = pool.uimap(squared, _x)
    _res2 = pool.uimap(busy_add, _x, _y, _d)
    assert sorted(_res1) == sorted(res1)
    assert sorted(_res2) == sorted(res2)

   #print(pool.amap)
    _res1 = pool.amap(squared, _x)
    _res2 = pool.amap(busy_add, _x, _y, _d)
    assert _res1.get() == res1
    assert _res2.get() == res2
   #print("")


def check_dill(pool, verbose=False): # test function that should fail in pickle
    if verbose:
        print(pool)
        print("x: %s\n" % str(x))

        print(pool.map.__name__)
   #start = time.time()
    try:
        res = pool.map(square_plus_one, x)
    except:
        assert False # should use a smarter test here...
   #end = time.time() - start
   #    print("time to results: %s" % end)
        print("y: %s\n" % str(res))
    assert True


def check_ready(pool, maxtries, delay, verbose=True):
    if verbose: print(pool)
    m = pool.amap(busy_squared, x)# x)

  # print(m.ready())
  # print(m.wait(0))
    tries = 0
    while not m.ready():
        time.sleep(delay)
        tries += 1
        if verbose: print("TRY: %s" % tries)
        if tries >= maxtries:
            if verbose: print("TIMEOUT")
            break
   #print(m.ready())
#   print(m.get(0))
    res = m.get()
    if verbose: print(res)
    z = [0]*len(x)
    assert res == list(map(squared, x))# x, z)
    assert tries > 0
    assert maxtries > tries #should be True, may not be if CPU is SLOW

def test_mp():
    from pathos.pools import ProcessPool as Pool
    pool = Pool(nodes=4)
    check_sanity( pool )
    check_maps( pool, items, delay )
    check_dill( pool )
    check_ready( pool, maxtries, delay, verbose=False )

def test_tp():
    from pathos.pools import ThreadPool as Pool
    pool = Pool(nodes=4)
    check_sanity( pool )
    check_maps( pool, items, delay )
    check_dill( pool )
    check_ready( pool, maxtries, delay, verbose=False )

def test_pp():
    from pathos.pools import ParallelPool as Pool
    pool = Pool(nodes=4)
    check_sanity( pool )
    check_maps( pool, items, delay )
    check_dill( pool )
    check_ready( pool, maxtries, delay, verbose=False )


if __name__ == '__main__':
    from pathos.helpers import freeze_support, shutdown
    freeze_support()
    test_mp()
    test_tp()
    test_pp()
    shutdown()
