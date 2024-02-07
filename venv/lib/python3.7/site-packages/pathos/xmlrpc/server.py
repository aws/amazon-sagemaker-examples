#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/pathos/blob/master/LICENSE
#
# adapted from J. Kim's XMLRPC server and request handler classes
"""
This module contains the base class for pathos XML-RPC servers,
and derives from python's SimpleXMLRPCServer, and the base class
for XML-RPC request handlers, which derives from python's base HTTP
request handler.


Usage
=====

A typical setup for an XML-RPC server will roughly follow this example:

    >>> # establish a XML-RPC server on a host at a given port
    >>> host = 'localhost'
    >>> port = 1234
    >>> server = XMLRPCServer(host, port)
    >>> print('port=%d' % server.port)
    >>>
    >>> # register a method the server can handle requests for
    >>> def add(x, y):
    ...     return x + y
    >>> server.register_function(add)
    >>>
    >>> # activate the callback methods and begin serving requests
    >>> server.activate()
    >>> server.serve()


The following is an example of how to make requests to the above server:

    >>> # establish a proxy connection to the server at (host,port)
    >>> host = 'localhost'
    >>> port = 1234
    >>> proxy = xmlrpclib.ServerProxy('http://%s:%d' % (host, port))
    >>> print('1 + 2 = %s' % proxy.add(1, 2))
    >>> print('3 + 4 = %s' % proxy.add(3, 4))

"""
__all__ = ['XMLRPCServer','XMLRPCRequestHandler']

import os
import socket
import xmlrpc.client as client
from http.server import BaseHTTPRequestHandler
from xmlrpc.server import SimpleXMLRPCDispatcher
from pathos.server import Server #XXX: pythia-0.6, was pyre.ipc.Server
from pathos.util import print_exc_info, spawn2, _str, _b
from pathos import logger


class XMLRPCServer(Server, SimpleXMLRPCDispatcher):
    '''extends base pathos server to an XML-RPC dispatcher'''

    def activate(self):
        """install callbacks"""
        
        Server.activate(self)
        self._selector.notifyOnReadReady(self._socket, self._onConnection)
        self._selector.notifyWhenIdle(self._onSelectorIdle)

        
    def serve(self):
        """enter the select loop... and wait for service requests"""
        
        timeout = 5
        Server.serve(self, 5)


    def _marshaled_dispatch(self, data, dispatch_method=None):
        """override SimpleXMLRPCDispatcher._marshaled_dispatch() fault string"""

        import xmlrpc.client as client
        from xmlrpc.client import Fault

        params, method = client.loads(data)

        # generate response
        try:
            if dispatch_method is not None:
                response = dispatch_method(method, params)
            else:
                response = self._dispatch(method, params)
            # wrap response in a singleton tuple
            response = (response,)
            response = client.dumps(response, methodresponse=1)
        except Fault as fault: # breaks 2.5 compatibility
            fault.faultString = print_exc_info()
            response = client.dumps(fault)
        except:
            # report exception back to server
            response = client.dumps(
                client.Fault(1, "\n%s" % print_exc_info())
                )

        return _b(response)


    def _registerChild(self, pid, fromchild):
        """register a child process so it can be retrieved on select events"""
        
        self._activeProcesses[fromchild] = pid
        self._selector.notifyOnReadReady(fromchild,
                                         self._handleMessageFromChild)


    def _unRegisterChild(self, fd):
        """remove a child process from active process register"""
        
        del self._activeProcesses[fd]


    def _handleMessageFromChild(self, selector, fd):
        """handler for message from a child process"""
        
        line = _str(fd.readline())
        if line[:4] == 'done':
            pid = self._activeProcesses[fd]
            os.waitpid(pid, 0)
        self._unRegisterChild(fd)


    def _onSelectorIdle(self, selector):
        '''something to do when there's no requests'''
        return True


    def _installSocket(self, host, port):
        """prepare a listening socket"""
        
        from pathos.portpicker import portnumber
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if port == 0: #Get a random port
            pick = portnumber(min=port, max=64*1024)
            while True:
                try:
                    port = pick()
                    s.bind((host, port))
                    break
                except socket.error:
                    continue
        else: #Designated port
            s.bind((host, port))
            
        s.listen(10)
        self._socket = s
        self.host = host
        self.port = port
        return
        
    def _onConnection(self, selector, fd):
        '''upon socket connection, establish a request handler'''
        if isinstance(fd, socket.SocketType):
            return self._onSocketConnection(fd)
        return None


    def _onSocketConnection(self, socket):
        '''upon socket connections, establish a request handler'''
        conn, addr = socket.accept()
        handler = XMLRPCRequestHandler(server=self, socket=conn)
        handler.handle()
        return True


    def __init__(self, host, port):
        '''create a XML-RPC server

Takes two initial inputs:
    host  -- hostname of XML-RPC server host
    port  -- port number for server requests
        '''
        Server.__init__(self)
        SimpleXMLRPCDispatcher.__init__(self,allow_none=False,encoding=None)

        self._installSocket(host, port)
        self._activeProcesses = {} #{ fd : pid }


class XMLRPCRequestHandler(BaseHTTPRequestHandler):
    ''' create a XML-RPC request handler '''

    _debug = logger(name="pathos.xmlrpc", level=30) # logging.WARN

    def do_POST(self):
        """ Access point from HTTP handler """

        
        def onParent(pid, fromchild, tochild):
            self._server._registerChild(pid, fromchild)
            tochild.write(_b('done\n'))
            tochild.flush()

        def onChild(pid, fromparent, toparent):
            try:
                response = self._server._marshaled_dispatch(data)
                self._sendResponse(response)
                line = _str(fromparent.readline())
                toparent.write(_b('done\n'))
                toparent.flush()
            except:
                logger(name='pathos.xmlrpc', level=30).error(print_exc_info())
            os._exit(0)

        try:
            data = self.rfile.read(int(self.headers['content-length']))
            params, method = client.loads(data)
            if method == 'run': #XXX: _str?
                return spawn2(onParent, onChild)
            else:
                response = self._server._marshaled_dispatch(data)
                self._sendResponse(response)
                return
        except:
            self._debug.error(print_exc_info())
            self.send_response(500)
            self.end_headers()
            return


    def log_message(self, format, *args):
        """ Overriding BaseHTTPRequestHandler.log_message() """

        self._debug.info("%s - - [%s] %s\n" %
                        (self.address_string(),
                         self.log_date_time_string(),
                         format%args))


    def _sendResponse(self, response):
        """ Write the XML-RPC response """

        response = _b(response)
        self.send_response(200)
        self.send_header("Content-type", "text/xml")
        self.send_header("Content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)
        self.wfile.flush()
        self.connection.shutdown(1)


    def __init__(self, server, socket):
        """
Override BaseHTTPRequestHandler.__init__(): we need to be able
to have (potentially) multiple handler objects at a given time.

Inputs:
    server  -- server object to handle requests for 
    socket  -- socket connection 
        """

        ## Settings required by BaseHTTPRequestHandler
        self.rfile = socket.makefile('rb', -1)
        self.wfile = socket.makefile('wb', 0)
        self.connection = socket
        self.client_address = (server.host, server.port)
        
        self._server = server


if __name__ == '__main__':
    pass


# End of file
