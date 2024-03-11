import re
import json


TRAINING_JOB_PREFIX_REGEX = "^[A-Za-z0-9\-]+$"
EMAIL_ADDRESS_REGEX = "^[a-z0-9]+[@]\w+[.]\w{2,3}$"
PHONE_NUMBER_REGEX = "^\+\d{1,15}$"


def validate_training_job_prefix(key, value):
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string!")
    if not re.match(TRAINING_JOB_PREFIX_REGEX, value):
        raise ValueError(
            "Invalid training job prefix! Must contain only letters, numbers and hyphens!"
        )


def validate_email_address(key, value):
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string!")
    if not re.match(EMAIL_ADDRESS_REGEX, value):
        raise ValueError("Invalid email address provided! Must follow this scheme: username@domain")


def validate_phone_number(key, value):
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string!")
    if not re.match(PHONE_NUMBER_REGEX, value):
        raise ValueError(
            """Invalid phone number provided! Must follow the E.164 format.
            See https://docs.aws.amazon.com/sns/latest/dg/sms_publish-to-phone.html for more info."""
        )


def validate_action_str(action_str, action_parameters):
    """
    Parse the action string as JSON within an exec call and verify that it matches the original action parameters.
    Note that we need the exec call to mimic the same behavior in the rules container.

    If this triggers a syntax error, the exec call is set up incorrectly and needs to be fixed.
    If this triggers a JSON decode error, the action string is badly formatted. This is probably due to invalid action
        parameters being specified (are you using any escape characters?)
    If this triggers an assertion error, the deserialized action JSON does not match the original action parameters,
        so the exec call is set up incorrectly and needs to be fixed.
    """
    try:
        exec(f'import json; assert json.loads("{action_str}") == {action_parameters}')
    except (SyntaxError, json.JSONDecodeError, AssertionError) as e:
        raise Exception(
            f"Error {type(e)} occurred during action string validation. See the docstring for more info."
        )
