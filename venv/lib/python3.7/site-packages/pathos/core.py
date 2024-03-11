#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
"""
high-level programming interface to core pathos utilities
"""
__all__ = ['copy', 'execute', 'kill', 'getpid', 'getppid', 'getchild', \
           'serve', 'connect', 'randomport']

import os
import string
import re
import pathos

# standard pattern for 'ps axj': '... ddddd ddddd ddddd ...'
_psaxj = re.compile(r"((\S+\s+)?\d+\s+\d+\s+\d+\s)")


def copy(source, destination=None, **kwds):
  '''copy source to (possibly) remote destination

Execute a copy, and return the copier. Use 'kill' to kill the copier, and 
'pid' to get the process id for the copier.

Args:
    source      -- path string of source 'file'
    destination -- path string for destination target
  '''
  #XXX: options, background, stdin can be set w/ kwds (also name, launcher)
  if destination is None: destination = os.getcwd()
  from pathos.secure import Copier
  opt = kwds.pop('options', None)
  kwds['background'] = kwds.pop('bg', False) # ignores 'background'
  copier = Copier(**kwds)
  if ':' in source or ':' in destination:
    if opt is None: opt = '-q -r'
    copier(options=opt, source=source, destination=destination)
  else:
    if opt is None: opt = '-r'
    copier(launcher='cp', options=opt, source=source, destination=destination)
  pathos.logger().info('executing {%s}', copier.message)
  copier.launch()
  copier.kill()
  return copier


def execute(command, host=None, bg=True, **kwds):
  '''execute a command (possibly) on a remote host

Execute a process, and return the launcher. Use 'response' to retrieve the
response from the executed command. Use 'kill' to kill the launcher, and 'pid'
to get the process id for the launcher.

Args:
    command -- command string to be executed
    host    -- hostname of execution target  [default = None (i.e. run locally)]
    bg      -- run as background process?  [default = True]
  '''
  #XXX: options, background, stdin can be set w/ kwds (also name, launcher)
  bg = bool(bg) # overrides 'background'
  if host in [None, '']:
    from pathos.connection import Pipe
    launcher = Pipe(**kwds)
    launcher(command=command, background=bg)
  else:
    from pathos.secure import Pipe
    opt = kwds.pop('options', '-q')
    launcher = Pipe(**kwds)
    launcher(options=opt, command=command, host=host, background=bg)
  pathos.logger().info('executing {%s}', launcher.message)
  launcher.launch()
 #response = launcher.response()
 #launcher.kill()
 #return response
  return launcher


#XXX: add local-only versions of kill and *pid to pox?
#XXX: use threading.Timer (or sched) to schedule or kill after N seconds?
def kill(pid, host=None, **kwds):
  '''kill a process (possibly) on a remote host

Args:
    pid   -- process id
    host  -- hostname where process is running [default = None (i.e. locally)]
  '''
  #XXX: launcher has "kill self" method; use it? note that this is different?
  command = 'kill -n TERM %s' % pid #XXX: use TERM=15 or KILL=9 ?
  getpid(pid, host) # throw error if pid doesn't exist #XXX: bad idea?
  response = execute(command, host, bg=False, **kwds).response()
  return response
  #XXX: how handle failed response?  bg=True prints, bg=False returns stderr


def _psax(response, pattern=None):
  """strips out bad lines in 'ps ax' response

  Takes multi-line string, response from execute('ps ax') or execute('ps axj').
  Takes an optional regex pattern for finding 'good' lines.  If pattern
  is None, assumes 'ps ax' was called.
  """
  if not response: return response
  if pattern:
    response = (line for line in response.split('\n') if pattern.match(line))
  else: # a 'ps ax' line should start with a 'digit'; " PID THING ..."
    response = (line for line in response.split('\n') \
                                 if line and line.lstrip()[0] in string.digits)
  return '\n'.join(response)


def getpid(target=None, host=None, all=False, **kwds):
  '''get the process id for a target process (possibly) running on remote host

This method should only be used as a last-ditch effort to find a process id.
This method __may__ work when a child has been spawned and the pid was not
registered... but there's no guarantee.

If target is None, then get the process id of the __main__  python instance.

Args:
    target -- string name of target process
    host   -- hostname where process is running
    all    -- get all resulting lines from query?  [default = False]
  '''
  if target is None:
    if all:
      target = ''
    elif host:
      raise OSError('[Error 3] No such process')
    else:
      return os.getpid()
  elif isinstance(target, int): #NOTE: passing pid useful for all=True
    target = "%5d " % target    #NOTE: assumes max pid is 99999
 #command = "ps -A | grep '%s'" % target # 'other users' only
  command = "ps ax | grep '%s'" % target # 'all users'
  response = _psax(execute(command, host, bg=False, **kwds).response())
  ignore = "grep %s" % target
  if all: return response

  try: # select the PID
    # find most recent where "grep '%s'" not in line
    pid = sorted(_select(line,(0,))[0] \
          for line in response.split('\n') if line and ignore not in line \
                                                   and command not in line)
    if pid is None:
      raise OSError('Failure to recover process id')
    #XXX: take advantage of *ppid to help match correct pid?
    return int(pid[-1])
  except (AttributeError, IndexError):
    raise OSError('[Error 3] No such process')


def _select(line, indx):
  '''select the correct data from the string, using the given index

  Takes a single string line, and a tuple of positional indicies.
  '''
  line = line.split()
  if max(indx) > len(line) - 1:
    return (None,None) # for the off chance there's a bad line
  return tuple(line[i] for i in indx)


def getppid(pid=None, host=None, group=False): # find parent of pid
  '''get parent process id (ppid) for the given process

If pid is None, the pid of the __main__  python instance will be used.

Args:
    pid    -- process id
    host   -- hostname where process is running
    group  -- get parent group id (pgid) instead of direct parent id?
  '''
  if pid is None:
    if host:
      raise OSError('[Error 3] No such process')
    return os.getpgrp() if group else os.getppid()
  pid = str(pid)
  command = "ps axj"
  response = execute(command, host).response()
  if response is None:
    raise OSError('[Errno 3] No such process')
  # analyze header for correct pattern and indx
  head = (line for line in response.split('\n') if 'PPID' in line)
  try:
    head = next(head).split()
  except StopIteration:
    raise OSError('Failure to recover process id')
  parent = 'PGID' if group else 'PPID'
  indx = (head.index('PID'), head.index(parent))
  # extract good data lines from response
  response = _psax(response, pattern=_psaxj)
  # select the PID and parent id
  response = dict(_select(line,indx) for line in response.split('\n') if line)
  response = response.get(pid, None)
  if response is None:
    raise OSError('[Errno 3] No such process')
  return int(response)


def getchild(pid=None, host=None, group=False): # find all children of pid
  '''get all child process ids for the given parent process id (ppid)

If pid is None, the pid of the __main__  python instance will be used.

Args:
    pid    -- parent process id
    host   -- hostname where process is running
    group  -- get process ids for the parent group id (pgid) instead?
  '''
  if pid is None:
    if host:
      raise OSError('[Error 3] No such process')
    pid = getpid()
  pid = str(pid)
  command = "ps axj"
  response = execute(command, host).response()
  if response is None:
    raise OSError('[Errno 3] No such process')
  # analyze header for correct pattern and indx
  head = (line for line in response.split('\n') if 'PPID' in line)
  try: head = next(head).split()
  except StopIteration:
    raise OSError('Failure to recover process id')
  parent = 'PGID' if group else 'PPID'
  indx = (head.index('PID'), head.index(parent))
  # extract good data lines from response
  response = _psax(response, pattern=_psaxj)
  # select the PID and parent id
  response = dict(_select(line,indx) for line in response.split('\n') if line)
  children = [int(key) for (key,value) in response.items() if value == pid]
  if children:
    return children
  if not group: # check to see if given 'PID' actually exists
    exists = [int(key) for (key,value) in response.items() if key == pid]
  else: exists = False # if 'PGID' not found, then doesn't exist
  if exists: return children
  raise OSError('[Errno 3] No such process')


def randomport(host=None):
  '''select a open port on a (possibly) remote host

Args:
    host -- hostname on which to select a open port
  '''
  from pathos.portpicker import randomport
  if not host:
    return randomport()
  from pathos.secure import Pipe
  from pathos.portpicker import __file__ as src
  # make sure src is a .py file, not .pyc or .pyo
  src = src.rstrip('co')
  launcher = Pipe() #XXX: use pox.which / which_python?
  launcher(command='python', host=host, background=False, stdin=open(src))
  pathos.logger().info('executing {python <%s} on %s', src, host)
  launcher.launch()
  try:
    rport = int(launcher.response())
  except:
    from pathos.secure import TunnelException
    raise TunnelException("failure to pick remote port")
  # return remote port number
  return rport


def connect(host, port=None, through=None):
  '''establish a secure tunnel connection to a remote host at the given port

Args:
    host     -- hostname to which a tunnel should be established
    port     -- port number (on host) to connect the tunnel to
    through  -- 'tunnel-through' hostname  [default = None]
  '''
  from pathos.secure import Tunnel
  t = Tunnel()
  t.connect(host, port, through)
  return t


#FIXME: needs work...
def serve(server, host=None, port=None, profile='.bash_profile'):
  '''begin serving RPC requests

Args:
    server: name of RPC server  (i.e. 'ppserver')
    host: hostname on which a server should be launched
    port: port number (on host) that server will accept request at
    profile: file to configure the user's environment [default='.bash_profile']
  '''
  if host is None: #XXX: and...?
    profile = ''
  else:
    profile = 'source %s; ' % profile
  file = '~/bin/%s.py' % server  #XXX: _should_ be on the $PATH
  if port is None: port = randomport(host)
  command = "%s -p %s" % (file,port)
  rserver = execute(command, host, bg=True)
  response = rserver.response()
  pathos.logger().info('response = %r', response)
  if response in ['', None]: #XXX: other responses allowed (?)
    pass
  else: #XXX: not really error checking...
    pathos.logger().error('invalid response = %r', response)
  from time import sleep
  delay = 2.0
  sleep(delay)
  return rserver


if __name__ == '__main__':
  pass

