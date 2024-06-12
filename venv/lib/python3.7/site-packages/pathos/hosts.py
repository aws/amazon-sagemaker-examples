#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
#
# dictionary of known host/profile pairs
"""
high-level programming interface to pathos host registry
"""

default_profile = '.bash_profile'
_profiles = { }
"""
For example, to register two 'known' host profiles:

  _profiles = { \
     'foobar.danse.us':'.profile', \
     'computer.cacr.caltech.edu':'.cshrc', \
  }
"""

def get_profile(rhost, assume=True):
  '''get the default $PROFILE for a remote host'''
  if rhost in _profiles:
    return _profiles[rhost]
  if assume:
    return default_profile
  return 


def get_profiles():
  '''get $PROFILE for each registered host'''
  return _profiles


def register_profiles(profiles):
  '''add dict of {'host':$PROFILE} to registered host profiles'''
  #XXX: needs parse checking of input
  _profiles.update(profiles)
  return


def register(rhost, profile=None):
  '''register a host and $PROFILE'''
  if profile == None:
    profile = default_profile
  #XXX: needs parse checking of input
  _profiles[rhost] = profile
  return 


# EOF
