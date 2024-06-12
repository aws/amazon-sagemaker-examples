#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE

from dill import source
def run_source(obj):
    _obj = source._wrap(obj)
    assert _obj(1.57) == obj(1.57)
    src = source.getimportable(obj, alias='_f')
    exec(src, globals())
    assert _f(1.57) == obj(1.57)
    name = source.getname(obj)
    assert name == obj.__name__ or src.split("=",1)[0].strip()

def run_ppmap(obj):
    from pathos.pools import ParallelPool
    p = ParallelPool(2)
    x = [1,2,3]
    assert list(map(obj, x)) == p.map(obj, x)
    p.clear()


def test_pp():
    from math import sin
    f = lambda x:x+1
    def g(x):
        return x+2

    for func in [g, f, abs, sin]:
        run_source(func)
        run_ppmap(func)


if __name__ == '__main__':
    test_pp()
