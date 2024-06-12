#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2022-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE

from pathos.maps import *
from pathos.pools import _ThreadPool as Pool

def squared(x):
    import time
    time.sleep(.1)
    return x*x

def added(*x):
    import time
    time.sleep(.1)
    return sum(x)


s = Map()
p = Map(Pool)
assert s(squared, range(4)) == p(squared, range(4))
del s, p

s = Imap()
p = Imap(Pool)
i = s(squared, range(4))
j = p(squared, range(4))
assert list(i) == list(j)
del s, p, i, j

s = Amap(Pool) #NotImplemented: Amap()
p = Uimap(Pool) #NotImplemented: Uimap()
assert sorted(p(squared, range(4))) == s(squared, range(4)).get()
del s, p


s = Smap()
p = Smap(Pool)
q = Asmap(Pool) #NotImplemented: Asmap()
sequence = [[0,1],[2,3],[4,5],[6,7]]
assert s(added, sequence) == p(added, sequence) == q(added, sequence).get()
del s, p, q

s = Ismap()
p = Ismap(Pool)
i = s(added, sequence)
j = p(added, sequence)
assert list(i) == list(j)
del s, p, i, j

p = Imap(Pool)
i = p(squared, range(4))
j = p(squared, range(4))
assert list(i) == list(j)
i = p(squared, range(4))
p.close()
p.join()
j = p(squared, range(4))
assert list(i) == list(j)
del p, i, j


import dill
s = Map(Pool)
p = Amap(Pool)
assert dill.copy(s)(squared, range(4)) == dill.copy(p)(squared, range(4)).get()
del s, p

import os
if not os.environ.get('COVERAGE'): #XXX: travis-ci
    from pathos.pools import _ProcessPool as _Pool
    s = Smap(Pool)
    p = Map(_Pool)
    r = s(p, [[squared, range(4)]]*4)
    del s, p

    s = Amap(Pool)
    p = Imap(_Pool)
    t = s(lambda x: list(p(squared, x)), [range(4)]*4)
    assert r == t.get()
    del s, p, r, t
