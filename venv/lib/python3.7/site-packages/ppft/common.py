# Parallel Python Software: http://www.parallelpython.com
# Copyright (c) 2005-2012 Vitalii Vanovschi.
# Copyright (c) 2015-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the author nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
"""
ppft common: a set of common utilities
"""

import threading
try: # for backward compatibilty
    import six
except ImportError:
    import types
    import sys
    six = types.ModuleType('six')
    six.PY3 = sys.version_info[0] == 3
    six.b = lambda x:x
    del types, sys
long = int
import io
file = io.IOBase
def str_(byte): # convert to unicode
    if not hasattr(byte, 'decode'): return byte
    try:
        return byte.decode('ascii')
    except UnicodeDecodeError: # non-ascii needs special handling
        return repr([i for i in byte])+'{B}'
def b_(string):
    if not hasattr(string, 'encode'): return string
    if not string.endswith(']{B}'): return string.encode('latin-1')
    return bytes(eval(string[:-3])) # special handling for non-ascii

# copyright, including original from Parallel Python
copyright = """Copyright (c) 2005-2012 Vitalii Vanovschi.
Copyright (c) 2015-2016 California Institute of Technology.
Copyright (c) 2016-2023 The Uncertainty Quantification Foundation."""
try: # the package is installed
    from .__info__ import __version__ as version
except: # pragma: no cover
    import os
    import sys
    parent = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    sys.path.append(parent)
    # get distribution meta info 
    from version import __version__ as version
    del os, sys, parent
__version__ = version = version.rstrip('.dev0') # release/target version only

def start_thread(name,  target,  args=(),  kwargs={},  daemon=True):
    """Starts a thread"""
    thread = threading.Thread(name=name,  target=target, args=args,  kwargs=kwargs)
    thread.daemon = daemon
    thread.start()
    return thread


def get_class_hierarchy(clazz):
    classes = []
    if clazz is type(object()):
        return classes
    for base_class in clazz.__bases__:
        classes.extend(get_class_hierarchy(base_class))
    classes.append(clazz)
    return classes


def is_not_imported(arg, modules):
    args_module = str(arg.__module__)
    for module in modules:
        if args_module == module or args_module.startswith(module + "."):
            return False
    return True


class portnumber(object):
    '''port selector

Usage:
    >>> pick = portnumber(min=1024,max=65535)
    >>> print( pick() )
    '''

    def __init__(self, min=0, max=64*1024):
        '''select a port number from a given range.

The first call will return a random number from the available range,
and each subsequent call will return the next number in the range.

Inputs:
    min -- minimum port number  [default = 0]
    max -- maximum port number  [default = 65536]
        '''
        self.min = min
        self.max = max
        self.first = -1
        self.current = -1
        return

    def __call__(self):
        import random
        
        if self.current < 0: #first call
            self.current = random.randint(self.min, self.max)
            self.first = self.current
            return self.current
        else:
            self.current += 1
            
            if self.current > self.max:
                self.current = self.min
            if self.current == self.first: 
                raise RuntimeError( 'Range exhausted' )
            return self.current
        return


def randomport(min=1024, max=65536):
    '''select a random port number

Inputs:
    min -- minimum port number  [default = 1024]
    max -- maximum port number  [default = 65536]
    '''
    return portnumber(min, max)()


# Parallel Python Software: http://www.parallelpython.com
