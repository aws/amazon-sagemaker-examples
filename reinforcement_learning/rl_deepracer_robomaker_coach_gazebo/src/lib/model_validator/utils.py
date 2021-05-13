import json
import os
import subprocess
import uuid

DEFAULT_AWS_CUSTOMER_ID = "unknown_customer_id"
DEFAULT_REQUEST_ID = "unknown_request_id"


def extract_custom_attributes(custom_attributes_string):
    """
    Extract session information from the header
    """
    session_info = dict()
    session_info["requestId"] = DEFAULT_REQUEST_ID
    session_info["awsAccountId"] = DEFAULT_AWS_CUSTOMER_ID
    session_info["traceId"] = str(uuid.uuid4())
    session_info["processId"] = os.getpid()

    if custom_attributes_string:
        custom_attributes = json.loads(custom_attributes_string)
        if "requestId" in custom_attributes:
            session_info["requestId"] = custom_attributes["requestId"]

        if "awsAccountId" in custom_attributes:
            session_info["awsAccountId"] = custom_attributes["awsAccountId"]

    return session_info


def run_cmd(
    cmd_args,
    change_working_directory="./",
    shell=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=None,
):
    """
    Function used to execute the shell commands

    Arguments:
        cmd_args (list): This is the list of commands to be appended in case
                         of shell=True, else executed as pipes.
        change_working_directory (string): This is to execute the command in different
                                           directory
        shell (bool): This is to say if the command to be executed as shell or not
        stdout (int): stdout stream target
        stderr (int): stderr stream target
        env (dict): environment variables
    Returns:
        (returncode, stdout, stderr) - tuples of returncode, stdout, and stderr
    """
    cmd = " ".join(map(str, cmd_args))
    process = subprocess.Popen(
        cmd if shell else cmd_args,
        cwd=change_working_directory,
        shell=shell,
        stdout=stdout,
        stderr=stderr,
        env=env,
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr
