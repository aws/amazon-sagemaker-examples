import json
import logging

from shared import lambda_context


def pack_and_rethrow_for_api_gateway(exception, context):
    """
    Packs an exception into a JSON object then throws a generic exception whose message contains this JSON object.
    In order to have the lambda runtime propagate the exception up to API Gateway, the exception thrown by this
    function must be unhandled.

    The exception should have these attributes set: error_type and http_status. If it does not, default values will be
    used instead.

    Credits for coming up with this pattern:
    https://aws.amazon.com/blogs/compute/error-handling-patterns-in-amazon-api-gateway-and-aws-lambda/

    :param exception: The exception to pack
    :type exception: Exception
    :param context: Lambda context
    :type context: object
    """

    # Error type for easy grepping. If the user received this default value the code threw an unhandled exception.
    # Each error type should be unique across the *entire* source tree. Treat it as if it were a filename/line combo.
    # ex. DSNNotFound, CMSConnectionTimeout, MalformedRequest
    error_type = "UnhandledException"

    # Status code the exception maps to. 500 by default
    http_status = 500

    token = lambda_context.get_token(context)

    # Human-readable
    error_string = str(exception)

    # Attempt to read exception metadata. If unable, default values from above will be used and a more verbose
    # stringification of the exception will be used
    if hasattr(exception, "error_type") and hasattr(exception, "http_status"):
        error_type = exception.error_type
        http_status = exception.http_status
    else:
        error_string = (
            "An exception has occurred that this service's maintainer did not catch. Please file a JIRA. "
            "Exception info: " + repr(exception)
        )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.error("This is the stack trace you're looking for:")
    logger.exception(exception)
    logger.error("Ignore the next line")

    # Finally, raise an exception that should be unhandled by this service's code. This will ensure that the exception
    # will be handled by the lambda runtime and the integration response mappings will be used to return
    # a non-200 HTTP status code to the user.
    raise Exception(
        json.dumps(
            {
                "httpStatus": http_status,
                "errorType": error_type,
                "errorString": error_string,
                "token": token,
            }
        )
    )


def raise_exception(http_status, error_type, message):
    """
    Raise an exception with metadata that the root lambda exception handler understands and will propagate up to the
    user.

    :param http_status: HTTP status code
    :type http_status: int
    :param error_type: Type of error. Should be unique across all source code
    :type error_type: str
    :param message: Human-readable representation of the error
    :type message: str
    """
    exception = Exception(message)
    exception.http_status = http_status
    exception.error_type = error_type
    raise exception
