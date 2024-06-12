#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Forked by: Mike McKerns (November 2004)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2004-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
"""
This module implements a selector class, which can be used to dispatch
events and for event handler wrangling.

"""

class Selector(object):
    """
Selector object for watching and event notification.
    """


    def watch(self, timeout=None):
        """dispatch events to the registered hanlders"""
        
        if timeout:
            self._timeout = timeout

        self._watch()
        return
    
        # FIXME:
        # leave like this until I understand better the set of exceptions I
        # would like to handle. It really is bad to catch all exceptions,
        # especially since it hides errors during development
        try:
            self._watch()

        except:
            # catch all exceptions
            self._cleanup()

            # get exception information
            import sys
            type, value = sys.exc_info()[:2]

            # rethrow the exception so the clients can handle it
            raise type(value)

        return


    def notifyOnReadReady(self, fd, handler):
        """add <handler> to the list of routines to call when <fd> is read ready"""
        self._input.setdefault(fd, []).append(handler)
        return


    def notifyOnWriteReady(self, fd, handler):
        """add <handler> to the list of routines to call when <fd> is write ready"""
        self._output.setdefault(fd, []).append(handler)
        return


    def notifyOnException(self, fd, handler):
        """add <handler> to the list of routines to call when <fd> raises an exception"""
        self._exception.setdefault(fd, []).append(handler)
        return


    def notifyOnInterrupt(self, handler):
        """add <handler> to the list of routines to call when a signal arrives"""
        self._interrupt.append(handler)
        return


    def notifyWhenIdle(self, handler):
        """add <handler> to the list of routines to call when a timeout occurs"""
        self._idle.append(handler)
        return


    def __init__(self):
        """
Takes no initial input.
        """
        self.state = True
        self._timeout = self._TIMEOUT

        # the fd activity clients
        self._input = {}
        self._output = {}
        self._exception = {}

        # clients to notify when there is nothing else to do
        self._idle = []
        self._interrupt = []
        
        return


    def _watch(self):
        import select

        while self.state:

            self._debug.debug("constructing list of watchers")
            iwtd = list(self._input.keys())
            owtd = list(self._output.keys())
            ewtd = list(self._exception.keys())

            self._debug.debug("input: %s" % iwtd)
            self._debug.debug("output: %s" % owtd)
            self._debug.debug("exception: %s" % ewtd)

            self._debug.debug("checking for indefinite block")
            if not iwtd and not owtd and not ewtd and not self._idle:
                self._debug.info("no registered handlers left; exiting")
                return

            self._debug.debug("calling select")
            try:
                reads, writes, excepts = select.select(iwtd, owtd, ewtd, self._timeout)
            except select.error as error: # breaks 2.5 compatibility
                # GUESS:
                # when a signal is delivered to a signal handler registered
                # by the application, the select call is interrupted and
                # raises a select.error
                errno, msg = error.args
                self._debug.info("signal received: %d: %s" % (errno, msg))
                continue
                
            self._debug.debug("returned from select")

            # dispatch to the idle handlers if this was a timeout
            if not reads and not writes and not excepts:
                self._debug.info("no activity; dispatching to idle handlers")
                for handler in self._idle:
                    if not handler(self):
                        self._idle.remove(handler)
            else:
                # dispatch to the registered handlers
                self._debug.info("dispatching to exception handlers")
                self._dispatch(self._exception, excepts)
                self._debug.info("dispatching to output handlers")
                self._dispatch(self._output, writes)
                self._debug.info("dispatching to input handlers")
                self._dispatch(self._input, reads)

        return


    def _dispatch(self, handlers, entities):

        for fd in entities:
            for handler in handlers[fd]:
                if not handler(self, fd):
                    handlers[fd].remove(handler)
            if not handlers[fd]:
                del handlers[fd]

        return


    def _cleanup(self):
        self._debug.info("cleaning up")
        for fd in self._input:
            fd.close()
        for fd in self._output:
            fd.close()
        for fd in self._exception:
            fd.close()

        for handler in self._interrupt:
            handler(self)

        return


    # static members
    from pathos import logger
    _debug = logger(name="pathos.selector", level=30) # logging.WARN
    del logger


    # constants
    _TIMEOUT = .5


#  End of file 
