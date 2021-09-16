# This is the file that implements a flask server to do inferences.
from gevent import monkey

monkey.patch_all()
import logging
import os
import shutil

import flask
from flask import json, request

from .response_handlers import ResponseHandler
from .utils import extract_custom_attributes, run_cmd

response_handler = ResponseHandler

# The logger object
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


MODEL_DATA_PATH = "/model_data"
S3_KMS_CMK_ARN_ENV = "S3_KMS_CMK_ARN_ENV"


# The flask app for serving inference
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy."""
    return response_handler.ping()


def build_validation_worker_cmd_args(s3_bucket, s3_prefix, aws_region):
    return [
        "python",
        "/opt/amazon/markov/validation_worker.py",
        "--s3_bucket",
        s3_bucket,
        "--s3_prefix",
        s3_prefix,
        "--aws_region",
        aws_region,
    ]


@app.route("/invocations", methods=["POST"])
def validate():
    """Validate user trained model located at given S3 location."""
    session_info = extract_custom_attributes(
        flask.request.headers.get("X-Amzn-SageMaker-Custom-Attributes")
    )
    if request.content_type != "application/json":
        return response_handler.unsupported_datatype(flask.request.content_type, session_info)

    logger.info(
        "The header info was extracted [header={}, session_info={}]".format(
            flask.request.headers, session_info
        )
    )

    request_args = request.get_json()
    logger.info(
        "Request received [session_info={}, content_type={}, args={}]".format(
            session_info, flask.request.content_type, request_args
        )
    )

    try:
        s3_bucket = request_args["s3_bucket"]
        s3_prefix = request_args["s3_prefix"]
        aws_region = request_args["aws_region"]
        subprocess_env = os.environ.copy()
        if "sse_key_id" in request_args and request_args["sse_key_id"]:
            subprocess_env[S3_KMS_CMK_ARN_ENV] = request_args["sse_key_id"]
            logger.info("S3_KMS_CMK_ARN_ENV set as {}".format(request_args["sse_key_id"]))
    except Exception as ex:
        return response_handler.argument_error(ex, session_info)

    custom_files_path = os.path.join(MODEL_DATA_PATH, session_info["traceId"])

    try:
        validator_cmd = build_validation_worker_cmd_args(
            s3_bucket=s3_bucket, s3_prefix=s3_prefix, aws_region=aws_region
        )
        logger.info(
            "Executing %s [session_info=%s]", " ".join(map(str, validator_cmd)), session_info
        )
        if not os.path.exists(custom_files_path):
            os.makedirs(custom_files_path)
        else:
            raise Exception("Custom Files Path already exists!: {}".format(custom_files_path))
        return_code, _, stderr = run_cmd(
            cmd_args=validator_cmd,
            change_working_directory=custom_files_path,
            shell=False,
            stdout=None,
            env=subprocess_env,
        )
        stderr = stderr.decode("utf-8")
        stderr_lines = stderr.splitlines()
        msg = "Validator exit with return code {} and stderr: {}".format(return_code, stderr)
        logger.info("%s [session_info=%s] ", msg, session_info)
        if return_code != 0:
            for line in stderr_lines:
                if "simapp_exception" in line:
                    err_json_string = line
                    err_json = json.loads(err_json_string)
                    if err_json["simapp_exception"]["errorCode"].startswith("4"):
                        return response_handler.invalid_model(err_json, session_info)
                    else:
                        return response_handler.simapp_error(err_json, session_info)
            raise Exception("Unhandled SimApp exception: {}".format(stderr))
        return response_handler.valid()
    except Exception as ex:
        logger.error(
            "Unknown Server Exception raised: {} [session_info={}]".format(ex, session_info)
        )
        return response_handler.server_error(ex, session_info)
    finally:
        # Make sure the temporary folder is deleted,
        # when validation_work preemptively exited with sys.exit
        # without deletion of temporary folder.
        shutil.rmtree(custom_files_path, ignore_errors=True)
