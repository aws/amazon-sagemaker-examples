import json
import logging

import flask

logger = logging.getLogger(__name__)


UNKNOWN_ERROR = "Transient error, Please try again."
UNSUPPORTED_TYPE_ERROR_FORMAT = (
    "This only supports application/json data. Unsupported data type: {}"
)
MISSING_REQUIRED_ARGUMENTS_ERROR = (
    "Missing required arguments (Required arguments: s3_bucket, s3_prefix, aws_region)."
)


class ResponseHandler(object):
    @staticmethod
    def ping():
        return flask.Response(response=json.dumps("pong"), status=200, mimetype="application/json")

    @staticmethod
    def valid():
        return flask.Response(response=json.dumps("valid"), status=200, mimetype="application/json")

    @staticmethod
    def unsupported_datatype(e=None, session_info=None):
        # 4xx errors
        logger.warning(
            "The content type, {}, is not supported. [session_info={}]".format(e, session_info)
        )
        message = UNSUPPORTED_TYPE_ERROR_FORMAT.format(e)
        return flask.Response(response=json.dumps(message), status=415, mimetype="application/json")

    @staticmethod
    def invalid_model(e=None, session_info=None):
        logger.warning("Invalid model. [session_info={}]".format(session_info), e)
        return flask.Response(response=json.dumps(e), status=400, mimetype="application/json")

    @staticmethod
    def simapp_error(e=None, session_info=None):
        logger.error(
            "An error occured during SimApp validation. [session_info={}]".format(session_info), e
        )
        return flask.Response(response=json.dumps(e), status=500, mimetype="application/json")

    @staticmethod
    def argument_error(e=None, session_info=None):
        logger.warning("Missing argument. [session_info={}]".format(session_info), e)
        return flask.Response(
            response=json.dumps(MISSING_REQUIRED_ARGUMENTS_ERROR),
            status=500,
            mimetype="application/json",
        )

    @staticmethod
    def server_error(e=None, session_info=None):
        # 5XX errors
        logger.error(
            "An error occurred during invoking sagemaker invocations API. [session_info={}]".format(
                session_info
            ),
            e,
            exc_info=True,
        )
        return flask.Response(
            response=json.dumps(UNKNOWN_ERROR), status=500, mimetype="application/json"
        )
