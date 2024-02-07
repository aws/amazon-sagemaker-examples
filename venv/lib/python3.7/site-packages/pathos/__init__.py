#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: June Kim (jkim @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE

# author, version, license, and long description
try: # the package is installed
    from .__info__ import __version__, __author__, __doc__, __license__
except: # pragma: no cover
    import os
    import sys
    parent = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    sys.path.append(parent)
    # get distribution meta info 
    from version import (__version__, __author__,
                         get_license_text, get_readme_as_rst)
    __license__ = get_license_text(os.path.join(parent, 'LICENSE'))
    __license__ = "\n%s" % __license__
    __doc__ = get_readme_as_rst(os.path.join(parent, 'README.md'))
    del os, sys, parent, get_license_text, get_readme_as_rst


# logger
def logger(level=None, handler=None, **kwds):
    """generate a logger instance for pathos

    Args:
        level (int, default=None): the logging level.
        handler (object, default=None): a ``logging`` handler instance.
        name (str, default='pathos'): name of the logger instance.
    Returns:
        configured logger instance.
    """
    import logging
    name = kwds.get('name', 'pathos')
    log = logging.getLogger(name)
    if handler is not None:
        log.handlers = []
        log.addHandler(handler)
    elif not len(log.handlers):
        log.addHandler(logging.StreamHandler())
    if level is not None:
        log.setLevel(level)
    return log

# high-level interface
from . import core
from . import hosts
from . import server
from . import selector
from . import connection
from . import pools
from . import maps

# worker pools
from . import serial
from . import parallel
from . import multiprocessing
from . import threading

# tools, utilities, etc
from . import util
from . import helpers

# backward compatibility
python = serial
pp = parallel
from pathos.secure import Pipe as SSH_Launcher
from pathos.secure import Copier as SCP_Launcher
from pathos.secure import Tunnel as SSH_Tunnel


def license():
    """print license"""
    print(__license__)
    return

def citation():
    """print citation"""
    print(__doc__[-491:-118])
    return

# end of file
