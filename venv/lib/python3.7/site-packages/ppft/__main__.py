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
ppft worker: a worker to communicate with ppserver
"""
import sys
import os
import io
ioStringIO = io.StringIO
try:
    import dill as pickle
except ImportError:
    import pickle
from . import transport as pptransport
from . import common as ppc

copyright = ppc.copyright
__version__ = version = ppc.__version__


def preprocess(msg):
    fname, fsources, imports = pickle.loads(ppc.b_(msg))
    fobjs = [compile(fsource, '<string>', 'exec') for fsource in fsources]
    for module in imports:
        try:            
            if not module.startswith("from ") and not module.startswith("import "):
                module = "import " + module
            exec(module)
            globals().update(locals())
        except:
            print("An error has occured during the module import")
            sys.excepthook(*sys.exc_info())
    return fname, fobjs

class _WorkerProcess(object):

    def __init__(self):
        self.hashmap = {}
        self.e = sys.__stderr__
        self.sout = ioStringIO()
#        self.sout = open("/tmp/pp.debug","a+")
        sys.stdout = self.sout
        sys.stderr = self.sout
        self.t = pptransport.CPipeTransport(sys.stdin, sys.__stdout__)
       #open('/tmp/pp.debug', 'a+').write('Starting _WorkerProcess\n')
       #open('/tmp/pp.debug', 'a+').write('send... \n')
        self.t.send(str(os.getpid()))
       #open('/tmp/pp.debug', 'a+').write('send: %s\n' % str(os.getpid()))
       #open('/tmp/pp.debug', 'a+').write('receive... \n')
        self.pickle_proto = int(self.t.receive())
       #open('/tmp/pp.debug', 'a+').write('receive: %s\n' % self.pickle_proto)

    def run(self):
        try:
            #execution cycle
            while 1:
                __fname, __fobjs = self.t.creceive(preprocess)

                __sargs = self.t.receive()

                for __fobj in __fobjs:
                    try:
                        exec(__fobj)
                        globals().update(locals())
                    except:
                        print("An error has occured during the " + \
                              "function import")
                        sys.excepthook(*sys.exc_info())

                __args = pickle.loads(ppc.b_(__sargs))
            
                __f = locals()[ppc.str_(__fname)]
                try:
                    __result = __f(*__args)
                except:
                    print("An error has occured during the function execution")
                    sys.excepthook(*sys.exc_info())
                    __result = None

                __sresult = pickle.dumps((__result, self.sout.getvalue()),
                        self.pickle_proto)

                self.t.send(__sresult)
                self.sout.truncate(0)
        except:
            print("A fatal error has occured during the function execution")
            sys.excepthook(*sys.exc_info())
            __result = None
            __sresult = pickle.dumps((__result, self.sout.getvalue()),
                    self.pickle_proto)
            self.t.send(__sresult)


if __name__ == "__main__":
        # add the directory with ppworker.py to the path
        sys.path.append(os.path.dirname(__file__))
        wp = _WorkerProcess()
        wp.run()

# Parallel Python Software: http://www.parallelpython.com
