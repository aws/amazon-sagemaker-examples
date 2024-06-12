#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
#
"""
``pathos`` interface to python's (serial) ``map`` functions

Notes:
    This module has been deprecated in favor of ``pathos.serial``.
"""
# backward compatibility
__all__ = ['PythonSerial']
from pathos.serial import __doc__, __STATE
from pathos.serial import *
PythonSerial = SerialPool

# EOF
