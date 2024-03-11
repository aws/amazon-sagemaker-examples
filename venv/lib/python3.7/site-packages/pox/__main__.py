#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pox/blob/master/LICENSE
"""
run any of the pox commands from the command shell prompt

Notes:
    - To get help, type ``$ pox`` at a shell terminal prompt.
    - For a list of available functions, type ``$ pox "help('pox')"``.
    - Incorrect function invocation will print the function's documentation.

Examples::

    $ pox "which('python')"
    /usr/bin/python
"""
from pox import *
from inspect import isfunction

def help(function=None):
    #XXX: better would be to provide a list of available commands
    if function == 'pox':
        print('Available functions:')
        print([key for (key,val) in globals().items() if isfunction(val) and not key.startswith('_')])
        return
    try:
        function = eval(function)
        if isfunction(function):
            print(function.__doc__)
            return
    except:
        pass
    print("Please provide a 'pox' command enclosed in quotes.\n")
    print("For example:")
    print("  $ pox \"which('python')\"")
    print("")
    help('pox')
    return


if __name__=='__main__':
    import sys
    try:
        func = sys.argv[1]
    except: func = None
    if func:
        try:
            exec('print(%s)' % func)
        except:
            print("Error: incorrect syntax '%s'\n" % func)
            exec('print(%s.__doc__)' % func.split('(')[0])
    else: help()


# End of file 
