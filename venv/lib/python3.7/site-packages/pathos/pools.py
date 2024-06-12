#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
"""
pools: pools of pathos workers, providing map and pipe constructs
"""

def _clear(type=None):
    "destroy all cached pools (of the given type)"
    pools = (ProcessPool, ThreadPool, ParallelPool, SerialPool)
    _pools = (_ProcessPool, _ThreadPool)
    #pools += _pools
    if type is None:
        for pool in pools:
            pool.__state__.clear()
    elif type in pools:
        type.__state__.clear()
    elif type in _pools:
        msg = "use the close() method to shutdown"
        raise NotImplementedError(msg)
    else:
        msg = "'%s' is not one of the pathos.pools" % type
        raise TypeError(msg)
    return


from pathos.helpers import ProcessPool as _ProcessPool
from pathos.helpers import ThreadPool as _ThreadPool
from pathos.multiprocessing import ProcessPool
from pathos.threading import ThreadPool
from pathos.parallel import ParallelPool
from pathos.serial import SerialPool

