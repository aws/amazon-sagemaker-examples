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
ppft transport: parallel python transport
"""

import logging
import os
import socket
import struct

import hashlib
sha_new = hashlib.sha1
md5_new = hashlib.md5

from . import common as ppc

copyright = ppc.copyright
__version__ = version = ppc.__version__

class Transport(object):

    def send(self, msg):
        raise NotImplemented("abstact function 'send' must be implemented "\
                "in a subclass")

    def receive(self, preprocess=None):
        raise NotImplemented("abstact function 'receive' must be implemented "\
                "in a subclass")

    def authenticate(self, secret):
        remote_version = ppc.str_(self.receive())
        if version != remote_version:
            logging.error("PP version mismatch (local: pp-%s, remote: pp-%s)"
                % (version, remote_version))
            logging.error("Please install the same version of PP on all nodes")
            return False
        srandom = ppc.b_(self.receive())
        secret = ppc.b_(secret)
        answer = sha_new(srandom+secret).hexdigest()
        self.send(answer)
        response = ppc.b_(self.receive())
        if response == ppc.b_("OK"):
            return True
        else:
            return False

    def close(self):
        pass

    def _connect(self, host, port):
        pass


class CTransport(Transport):
    """Cached transport
    """
    rcache = {}

    def hash(self, msg):
        return md5_new(ppc.b_(msg)).hexdigest()

    def csend(self, msg):
       #if hasattr(self, 'w'):
       #    open('/tmp/pp.debug', 'a+').write(repr(('cs', self.w, msg))+'\n')
       #else:
       #    open('/tmp/pp.debug', 'a+').write(repr(('cs', self.socket, msg))+'\n')
        msg = ppc.b_(msg)
        hash1 = self.hash(msg)
        if hash1 in self.scache:
            self.send(ppc.b_("H" + hash1))
        else:
            self.send(ppc.b_("N") + msg)
            self.scache[hash1] = True

    def creceive(self, preprocess=None):
        msg = self.receive()
       #if hasattr(self, 'r'):
       #    open('/tmp/pp.debug', 'a+').write(repr(('cr',  self.r, msg))+'\n')
       #else:
       #    open('/tmp/pp.debug', 'a+').write(repr(('cr', self.socket, msg))+'\n')
        msg = ppc.b_(msg)
        if msg[:1] == ppc.b_('H'):
            hash1 = ppc.str_(msg[1:])
        else:
            msg = msg[1:]
            hash1 = self.hash(msg)
            if preprocess is None: preprocess = lambda x:x
            self.rcache[hash1] = tuple(map(preprocess, (msg, )))[0]
        return self.rcache[hash1]


class PipeTransport(Transport):

    def __init__(self, r, w):
       #open('/tmp/pp.debug', 'a+').write(repr((r,w))+'\n')
        self.scache = {}
        self.exiting = False
        if isinstance(r, ppc.file) and isinstance(w, ppc.file):
            self.r = r
            self.w = w
        else:
            raise TypeError("Both arguments of PipeTransport constructor " \
                    "must be file objects")
        if hasattr(self.w, 'buffer'):
            self.wb = self.w.buffer
            self.has_wb = True
        else:
            self.wb = self.w
            self.has_wb = False
        if hasattr(self.r, 'buffer'):
            self.rb = self.r.buffer
            self.has_rb = True
        else:
            self.rb = self.r
            self.has_rb = False
        

    def send(self, msg):
       #l = len(ppc.b_(msg)) if (self.has_wb or self.w.mode == 'wb') else len(ppc.str_(msg))
       #open('/tmp/pp.debug', 'a+').write(repr(('s', l, self.w, msg))+'\n')
        if self.has_wb or self.w.mode == 'wb':
            msg = ppc.b_(msg)
            self.wb.write(struct.pack("!Q", len(msg)))
            self.w.flush()
        else: #HACK: following may be > 8 bytes, needed for len(msg) >= 256
            msg = ppc.str_(msg)
            self.wb.write(ppc.str_(struct.pack("!Q", len(msg))))
            self.w.flush()
        self.wb.write(msg)
        self.w.flush()

    def receive(self, preprocess=None):
        e_size = struct.calcsize("!Q") # 8
        c_size = struct.calcsize("!c") # 1
        r_size = 0
        stub = ppc.b_("") if (self.has_rb or self.r.mode == 'rb') else ""
        data = stub
        while r_size < e_size:
            msg = self.rb.read(e_size-r_size)
           #l = len(msg)
           #open('/tmp/pp.debug', 'a+').write(repr(('_r', l, self.r, msg))+'\n')
            if msg == stub:
                raise RuntimeError("Communication pipe read error")
            if stub == "" and msg.startswith('['): #HACK to get str_ length
                while not msg.endswith('{B}'):
                    msg += self.rb.read(c_size)
            r_size += len(msg)
            data += msg
        e_size = struct.unpack("!Q", ppc.b_(data))[0] # get size of msg

        r_size = 0
        data = stub
        while r_size < e_size:
            msg = self.rb.read(e_size-r_size)
           #l = len(msg)
           #open('/tmp/pp.debug', 'a+').write(repr(('r_', l, self.r, msg))+'\n')
            if msg == stub:
                raise RuntimeError("Communication pipe read error")
            r_size += len(msg)
            data += msg
        data = ppc.b_(data)

        if preprocess is None: preprocess = lambda x:x
        return tuple(map(preprocess, (data, )))[0]

    def close(self):
        self.w.close()
        self.r.close()


class SocketTransport(Transport):

    def __init__(self, socket1, socket_timeout):
        if socket1:
            self.socket = socket1
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(socket_timeout)
        self.scache = {}

    def send(self, data):
       #l = len(ppc.b_(data))
       #open('/tmp/pp.debug', 'a+').write(repr(('ss', l, self.socket, data))+'\n')
        data = ppc.b_(data)
        size = struct.pack("!Q", len(data))
        t_size = struct.calcsize("!Q")
        s_size = ppc.long(0)
        while s_size < t_size:
            p_size = self.socket.send(size[s_size:])
            if p_size == 0:
                raise RuntimeError("Socket connection is broken")
            s_size += p_size

        t_size = len(data)
        s_size = ppc.long(0)
        while s_size < t_size:
            p_size = self.socket.send(data[s_size:])
            if p_size == 0:
                raise RuntimeError("Socket connection is broken")
            s_size += p_size

    def receive(self, preprocess=None):
        e_size = struct.calcsize("!Q")
        r_size = 0
        stub = ppc.b_("")
        data = stub
        while r_size < e_size:
            msg = self.socket.recv(e_size-r_size)
           #l = len(msg)
           #open('/tmp/pp.debug', 'a+').write(repr(('_sr', l, self.socket, msg))+'\n')
            if msg == stub:
                raise RuntimeError("Socket connection is broken")
            r_size += len(msg)
            data += msg
        e_size = struct.unpack("!Q", ppc.b_(data))[0] # get size of msg

        r_size = 0
        data = stub
        while r_size < e_size:
            msg = self.socket.recv(e_size-r_size)
           #l = len(msg)
           #open('/tmp/pp.debug', 'a+').write(repr(('sr_', l, self.socket, msg))+'\n')
            if msg == stub:
                raise RuntimeError("Socket connection is broken")
            r_size += len(msg)
            data += msg
        data = ppc.b_(data)
        return data

    def close(self):
        self.socket.close()

    def _connect(self, host, port):
        self.socket.connect((host, port))


class CPipeTransport(PipeTransport, CTransport):
    pass


class CSocketTransport(SocketTransport, CTransport):
    pass
    
# Parallel Python Software: http://www.parallelpython.com
