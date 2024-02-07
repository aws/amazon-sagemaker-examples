#!/usr/bin/env python
#
# Authors: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
# Copyright (c) 2010 Gael Varoquaux
# License: BSD Style, 3 clauses.

# Forked by: Mike McKerns (December 2013)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2013-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pox/blob/master/LICENSE
"""
Disk management utilities.
"""

import os
import shutil
import sys
import time


def disk_used(path):
    """get the disk usage for the given directory

    Args:
        path (str): path string.

    Returns:
        int corresponding to disk usage in blocks.
    """
    size = 0
    for file in os.listdir(path) + ['.']:
        stat = os.stat(os.path.join(path, file))
        if hasattr(stat, 'st_blocks'):
            size += stat.st_blocks * 512
        else:
            # on some platform st_blocks is not available (e.g., Windows)
            # approximate by rounding to next multiple of 512
            size += (stat.st_size // 512 + 1) * 512
    # We need to convert to int to avoid having longs on some systems (we
    # don't want longs to avoid problems we SQLite)
    return int(size / 1024.)


def kbytes(text):
    """convert memory text to the corresponding value in kilobytes

    Args:
        text (str): string corresponding to an abbreviation of size.

    Returns:
        int representation of text.

    Examples:
        >>> kbytes(\'10K\')
        10
        >>> 
        >>> kbytes(\'10G\')
        10485760
    """
    kilo = 1024
    units = dict(K=1, M=kilo, G=kilo ** 2)
    try:
        size = int(units[text[-1]] * float(text[:-1]))
    except (KeyError, ValueError):
        raise ValueError(
                "Invalid literal for size: '%s' should be "
                "a string like '10G', '500M', '50K'" % text
                )
    return size


# if a rmtree operation fails, wait for this much time (in secs),
# then retry once. if it still fails, raise the exception
RM_SUBDIRS_RETRY_TIME = 0.1

def rmtree(path, self=True, ignore_errors=False, onerror=None):
    """remove directories in the given path

    Args:
        path (str): path string of root of directories to delete.
        self (bool, default=True): if False, delete subdirectories, not path.
        ignore_errors (bool, default=False): if True, silently ignore errors.
        onerror (function, default=None): custom error handler.

    Returns:
        None

    Notes:
        If self=False, the directory indicated by path is left in place,
        and its subdirectories are erased. If self=True, path is also removed.

        If ignore_errors=True, errors are ignored. Otherwise, onerror is called
        to handle the error with arguments ``(func, path, exc_info)``, where
        *func* is ``os.listdir``, ``os.remove``, or ``os.rmdir``; *path* is the
        argument to the function that caused it to fail; and *exc_info* is a
        tuple returned by ``sys.exc_info()``. If ignore_errors=False and
        onerror=None, an exception is raised.
    """
    names = []
    try:
        names = os.listdir(path)
    except os.error:
        if onerror is not None:
            onerror(os.listdir, path, sys.exc_info())
        elif ignore_errors:
            return
        else:
            raise
    if self:
        names = ['']

    for name in names:
        fullname = os.path.join(path, name)
        if os.path.isdir(fullname):
            if onerror is not None:
                shutil.rmtree(fullname, ignore_errors, onerror)
            else:
                # allow the rmtree to fail once, wait and re-try.
                # if the error is raised again, fail
                err_count = 0
                while True:
                    try:
                        shutil.rmtree(fullname, ignore_errors, None)
                        break
                    except os.error:
                        if err_count > 0:
                            raise
                        err_count += 1
                        time.sleep(RM_SUBDIRS_RETRY_TIME)


# EOF
