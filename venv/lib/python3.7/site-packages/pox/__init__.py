#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pox/blob/master/LICENSE

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


from .shutils import shelltype, homedir, rootdir, username, sep, \
                     minpath, env, whereis, which, find, walk, where, \
                     mkdir, rmtree, shellsub
from .utils import pattern, expandvars, getvars, convert, replace, select, \
                   findpackage, wait_for, disk_used, remote, which_python, \
                   parse_remote, kbytes, selectdict, index_slice, index_join


def license():
    """print the license"""
    print(__license__)
    return

def citation():
    """print the citation"""
    print(__doc__[-491:-118])
    return

# end of file
