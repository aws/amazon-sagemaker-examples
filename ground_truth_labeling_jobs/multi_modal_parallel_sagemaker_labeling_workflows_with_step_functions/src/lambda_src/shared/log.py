import json
import logging
import sys
import time
import traceback
from pprint import pformat

# Root-level loggers should only print when something bad happens
logger = logging.getLogger()
logger.setLevel(logging.WARN)


#
# Helpers
#


def log_request_and_context(request, context):
    logger.debug("Request:\n" + pformat(request, indent=4))
    logger.debug("Lambda context:\n" + pformat(vars(context), indent=4))


def log_response(response):
    logger.debug("Response:\n" + pformat(response, indent=4))


#
# Internal stuff
#

# The logger that is exported from this package will have the tag 'smgt'
logger = logging.getLogger("smgt")
logger.setLevel(logging.DEBUG)
