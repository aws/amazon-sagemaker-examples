#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
"""
demonstrates pickle/source failure cases with decorators/factories and pp
"""

def __wrap_nested(function, inner_function):
    def function_wrapper(x):
        _x = x[:]
        return function(inner_function(_x))
    return function_wrapper

def wrap_nested(inner_function):
    def dec(function):
        def function_wrapper(x):
            _x = x[:]
            return function(inner_function(_x))
        return function_wrapper
    return dec

class _wrap_nested(object):
    def __init__(self, inner_function):
        self._inner_function = inner_function
    def __call__(self, function):
        def function_wrapper(x):
            _x = x[:]
            return function(self._inner_function(_x))
        return function_wrapper

def add(*args):
    #from numpy import sum
    return sum(*args)

'''                               # FAILS to find 'add' (returns [None,None])
@wrap_nested(add)
def addabs(x):
    return abs(x)
'''

#addabs = __wrap_nested(abs, add) # ok
wrapadd = wrap_nested(add)        # ok
#wrapadd = _wrap_nested(add)      # HANGS
addabs = wrapadd(abs)             # <required for the latter two above>
#'''


def test_wrap():
    x = [(-1,-2),(3,-4)]
    y = [3, 1]
    assert list(map(addabs, x)) == y

    from pathos.pools import ProcessPool as Pool
    assert Pool().map(addabs, x) == y

    from pathos.pools import ParallelPool as Pool
    assert Pool().map(addabs, x) == y


if __name__ == '__main__':
    from pathos.helpers import freeze_support, shutdown
    freeze_support()
    test_wrap()
    shutdown()
