#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2015-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/ppft/blob/master/LICENSE
from ppft.server.__main__ import *
from ppft.server.__main__ import __version__, _NetworkServer, __doc__


if __name__ == "__main__":
    server = create_network_server(sys.argv[1:])
    statsignal = getattr(signal, STAT_SIGNAL, None)
    if statsignal:
        signal.signal(statsignal, signal_handler)
    server.listen()
    #have to destroy it here explicitly otherwise an exception
    #comes out in Python 2.4
    del server


# EOF
