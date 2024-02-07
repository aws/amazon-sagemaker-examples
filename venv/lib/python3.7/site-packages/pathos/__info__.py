#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
'''
--------------------------------------------------------------------------
pathos: parallel graph management and execution in heterogeneous computing
--------------------------------------------------------------------------

About the Pathos Framework
==========================

``pathos`` is a framework for heterogeneous computing. It provides a consistent
high-level interface for configuring and launching parallel computations
across heterogeneous resources. ``pathos`` provides configurable launchers for
parallel and distributed computing, where each launcher contains the
syntactic logic to configure and launch jobs in an execution environment.
Examples of launchers that plug into ``pathos`` are: a queue-less MPI-based
launcher (in ``pyina``), a ssh-based launcher (in ``pathos``), and a multi-process
launcher (in ``multiprocess``).

``pathos`` provides a consistent interface for parallel and/or distributed
versions of ``map`` and ``apply`` for each launcher, thus lowering the barrier
for users to extend their code to parallel and/or distributed resources.
The guiding design principle behind ``pathos`` is that ``map`` and ``apply``
should be drop-in replacements in otherwise serial code, and thus switching
to one or more of the ``pathos`` launchers is all that is needed to enable
code to leverage the selected parallel or distributed computing resource.
This not only greatly reduces the time to convert a code to parallel, but it
also enables a single code-base to be maintained instead of requiring
parallel, serial, and distributed versions of a code. ``pathos`` maps can be
nested, thus hierarchical heterogeneous computing is possible by merely
selecting the desired hierarchy of ``map`` and ``pipe`` (``apply``) objects.

The ``pathos`` framework is composed of several interoperating packages:

    - ``dill``: serialize all of Python
    - ``pox``: utilities for filesystem exploration and automated builds
    - ``klepto``: persistent caching to memory, disk, or database
    - ``multiprocess``: better multiprocessing and multithreading in Python
    - ``ppft``: distributed and parallel Python
    - ``pyina``: MPI parallel ``map`` and cluster scheduling
    - ``pathos``: graph management and execution in heterogeneous computing


About Pathos
============

The ``pathos`` package provides a few basic tools to make parallel and
distributed computing more accessible to the end user. The goal of ``pathos``
is to enable the user to extend their own code to parallel and distributed
computing with minimal refactoring.

``pathos`` provides methods for configuring, launching, monitoring, and
controlling a service on a remote host. One of the most basic features
of ``pathos`` is the ability to configure and launch a RPC-based service
on a remote host. ``pathos`` seeds the remote host with the  ``portpicker``
script, which allows the remote host to inform the localhost of a port
that is available for communication.

Beyond the ability to establish a RPC service, and then post requests,
is the ability to launch code in parallel. Unlike parallel computing
performed at the node level (typically with MPI), ``pathos`` enables the
user to launch jobs in parallel across heterogeneous distributed resources.
``pathos`` provides distributed ``map`` and ``pipe`` algorithms, where a mix of
local processors and distributed workers can be selected.  ``pathos``
also provides a very basic automated load balancing service, as well as
the ability for the user to directly select the resources.

The high-level ``pool.map`` interface, yields a ``map`` implementation that
hides the RPC internals from the user. With ``pool.map``, the user can launch
their code in parallel, and as a distributed service, using standard Python
and without writing a line of server or parallel batch code.

RPC servers and communication in general is known to be insecure.  However,
instead of attempting to make the RPC communication itself secure, ``pathos``
provides the ability to automatically wrap any distributes service or
communication in a ssh-tunnel. Ssh is a universally trusted method.
Using ssh-tunnels, ``pathos`` has launched several distributed calculations
on national lab clusters, and to date has performed test calculations
that utilize node-to-node communication between several national lab clusters
and a user's laptop.  ``pathos`` allows the user to configure and launch
at a very atomistic level, through raw access to ssh and scp. 

``pathos`` is the core of a Python framework for heterogeneous computing.
``pathos`` is in active development, so any user feedback, bug reports, comments,
or suggestions are highly appreciated.  A list of issues is located at https://github.com/uqfoundation/pathos/issues, with a legacy list maintained at https://uqfoundation.github.io/project/pathos/query.


Major Features
==============

``pathos`` provides a configurable distributed parallel ``map`` interface
to launching RPC service calls, with:

    - a ``map`` interface that meets and extends the Python ``map`` standard
    - the ability to submit service requests to a selection of servers
    - the ability to tunnel server communications with ssh

The ``pathos`` core is built on low-level communication to remote hosts using
ssh. The interface to ssh, scp, and ssh-tunneled connections can:

    - configure and launch remote processes with ssh
    - configure and copy file objects with scp
    - establish an tear-down a ssh-tunnel

To get up and running quickly, ``pathos`` also provides infrastructure to:

    - easily establish a ssh-tunneled connection to a RPC server


Current Release
===============

The latest released version of ``pathos`` is available from:

    https://pypi.org/project/pathos

``pathos`` is distributed under a 3-clause BSD license.


Development Version
===================

You can get the latest development version with all the shiny new features at:

    https://github.com/uqfoundation

If you have a new contribution, please submit a pull request.


Installation
============

``pathos`` can be installed with ``pip``::

    $ pip install pathos


Requirements
============

``pathos`` requires:

    - ``python`` (or ``pypy``), **>=3.7**
    - ``setuptools``, **>=42**
    - ``pox``, **>=0.3.3**
    - ``dill``, **>=0.3.7**
    - ``ppft``, **>=1.7.6.7**
    - ``multiprocess``, **>=0.70.15**


More Information
================

Probably the best way to get started is to look at the documentation at
http://pathos.rtfd.io. Also see ``pathos.tests`` and https://github.com/uqfoundation/pathos/tree/master/examples for a set of scripts that demonstrate the
configuration and launching of communications with ssh and scp, and demonstrate
the configuration and execution of jobs in a hierarchical parallel workflow.
You can run the test suite with ``python -m pathos.tests``. Tunnels and other
connections to remote servers can be established with the ``pathos_connect``
script (or with ``python -m pathos``). See ``pathos_connect --help`` for more
information.  ``pathos`` also provides a ``portpicker`` script to select an
open port (also available with ``python -m pathos.portpicker``). The source
code is generally well documented, so further questions may be resolved by
inspecting the code itself.  Please feel free to submit a ticket on github,
or ask a question on stackoverflow (**@Mike McKerns**). If you would like to
share how you use ``pathos`` in your work, please send an email (to **mmckerns
at uqfoundation dot org**).

Important classes and functions are found here:

    - ``pathos.abstract_launcher``           [the worker pool API definition]
    - ``pathos.pools``                       [all of the pathos worker pools]
    - ``pathos.core``                        [the high-level command interface] 
    - ``pathos.hosts``                       [the hostname registry interface] 
    - ``pathos.serial.SerialPool``           [the serial Python worker pool]
    - ``pathos.parallel.ParallelPool``       [the parallelpython worker pool]
    - ``pathos.multiprocessing.ProcessPool`` [the multiprocessing worker pool]
    - ``pathos.threading.ThreadPool``        [the multithreading worker pool]
    - ``pathos.connection.Pipe``             [the launcher base class]
    - ``pathos.secure.Pipe``                 [the secure launcher base class]
    - ``pathos.secure.Copier``               [the secure copier  base class]
    - ``pathos.secure.Tunnel``               [the secure tunnel base class]
    - ``pathos.selector.Selector``           [the selector base class]
    - ``pathos.server.Server``               [the server base class]
    - ``pathos.profile``                     [profiling in threads and processes]
    - ``pathos.maps``                        [standalone map instances]

``pathos`` also provides two convenience scripts that are used to establish
secure distributed connections. These scripts are installed to a directory
on the user's ``$PATH``, and thus can be run from anywhere:

    - ``portpicker``                         [get the portnumber of an open port]
    - ``pathos_connect``                     [establish tunnel and/or RPC server]

Typing ``--help`` as an argument to any of the above scripts will print out an
instructive help message.


Citation
========

If you use ``pathos`` to do research that leads to publication, we ask that you
acknowledge use of ``pathos`` by citing the following in your publication::

    M.M. McKerns, L. Strand, T. Sullivan, A. Fang, M.A.G. Aivazis,
    "Building a framework for predictive science", Proceedings of
    the 10th Python in Science Conference, 2011;
    http://arxiv.org/pdf/1202.1056

    Michael McKerns and Michael Aivazis,
    "pathos: a framework for heterogeneous computing", 2010- ;
    https://uqfoundation.github.io/project/pathos

Please see https://uqfoundation.github.io/project/pathos or
http://arxiv.org/pdf/1202.1056 for further information.

'''

__version__ = '0.3.1'
__author__ = 'Mike McKerns'

__license__ = '''
Copyright (c) 2004-2016 California Institute of Technology.
Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
All rights reserved.

This software is available subject to the conditions and terms laid
out below. By downloading and using this software you are agreeing
to the following conditions.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    - Neither the names of the copyright holders nor the names of any of
      the contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
