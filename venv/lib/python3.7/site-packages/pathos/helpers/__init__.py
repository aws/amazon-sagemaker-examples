#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE

from . import pp_helper
from . import mp_helper
import ppft as parallelpython

try:
    import multiprocess as mp
    from multiprocess.pool import Pool as ProcessPool
    from multiprocess import cpu_count
    from multiprocess.dummy import Pool as ThreadPool
    from multiprocess import freeze_support
except ImportError:  # fall-back to package distributed with python
    import multiprocessing as mp
    from multiprocessing.pool import Pool as ProcessPool
    from multiprocessing import cpu_count
    from multiprocessing.dummy import Pool as ThreadPool
    from multiprocessing import freeze_support

from pathos.pools import _clear as shutdown
