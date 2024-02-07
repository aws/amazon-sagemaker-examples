#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
"""
map helper functions
"""
# random_state and random_seed copied from mystic.tools

def starargs(f):
    """decorator to convert a many-arg function to a single-arg function"""
    func = lambda args: f(*args)
   #func.__module__ = f.__module__
   #func.__name__ = f.__name__
    doc = "\nNOTE: all inputs have been compressed into a single argument"
    if f.__doc__: func.__doc__ = f.__doc__ + doc
    return func
   #from functools import update_wrapper
   #return update_wrapper(func, f)


def random_seed(s=None):
    "sets the seed for calls to 'random()'"
    import random
    random.seed(s)
    try:
        from numpy import random
        random.seed(s)
    except:
        pass
    return


def random_state(module='random', new=False, seed='!'):
    """return a (optionally manually seeded) random generator

For a given module, return an object that has random number generation (RNG)
methods available.  If new=False, use the global copy of the RNG object.
If seed='!', do not reseed the RNG (using seed=None 'removes' any seeding).
If seed='*', use a seed that depends on the process id (PID); this is useful
for building RNGs that are different across multiple threads or processes.
    """
    import random
    if module == 'random':
        rng = random
    elif not isinstance(module, type(random)):
        # convienence for passing in 'numpy'
        if module == 'numpy': module = 'numpy.random'
        try:
            import importlib
            rng = importlib.import_module(module)
        except ImportError:
            rng = __import__(module, fromlist=module.split('.')[-1:])
    elif module.__name__ == 'numpy': # convienence for passing in numpy
        from numpy import random as rng
    else: rng = module

    _rng = getattr(rng, 'RandomState', None) or \
           getattr(rng, 'Random') # throw error if no rng found
    if new:
        rng = _rng()

    if seed == '!': # special case: don't reset the seed
        return rng
    if seed == '*': # special case: random seeding for multiprocessing
        try:
            import multiprocessing as mp
            seed = mp.current_process().pid
        except:
            seed = 0
        import time
        seed += int(time.time()*1e6)

    # set the random seed (or 'reset' with None)
    rng.seed(seed)
    return rng

