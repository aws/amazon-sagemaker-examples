from smdebug_rulesconfig import ActionList, StopTraining, Email, SMS, is_valid_action_object

import pytest


@pytest.fixture
def stop_training_name():
    return StopTraining.__name__.lower()


@pytest.fixture
def email_name():
    return Email.__name__.lower()


@pytest.fixture
def sms_name():
    return SMS.__name__.lower()


@pytest.fixture
def training_job_prefix():
    return "training-job-prefix"


@pytest.fixture
def training_job_name():
    return "training-job-name"


@pytest.fixture
def email_address():
    return "abc@abc.com"


@pytest.fixture
def phone_number():
    return "+1234567890"


def test_default_stop_training_action(stop_training_name, training_job_name):
    """
    Validate that creating a `StopTraining` action object is a valid action object, that it is serialized correctly into
    an action string, and that the resulting action parameters are correct.

    Also validate that creating this object without a custom training job prefix allows this prefix to be manually
    updated later on (this mimics the exact behavior in sagemaker SDK, where the actual training job name is used
    as the prefix if the user did not specify a custom training job prefix).
    """
    action = StopTraining()
    assert is_valid_action_object(action)
    assert action.use_default_training_job_prefix is True
    assert action.action_parameters == {"name": stop_training_name}

    action.update_training_job_prefix_if_not_specified(training_job_name)
    assert action.action_parameters == {
        "name": stop_training_name,
        "training_job_prefix": training_job_name,
    }


def test_custom_stop_training_action(stop_training_name, training_job_prefix, training_job_name):
    """
    Validate that creating a `StopTraining` action object is a valid action object, that it is serialized correctly into
    an action string, and that the resulting action parameters are correct.

    Also validate that creating this object with a custom training job prefix does not allow this prefix to be
    manually updated later on (this again mimics the exact behavior in sagemaker SDK: if user specifies custom prefix,
    this should actually be used in the rule container).
    """
    action = StopTraining(training_job_prefix)
    assert is_valid_action_object(action)
    assert action.use_default_training_job_prefix is False
    assert action.action_parameters == {
        "name": stop_training_name,
        "training_job_prefix": training_job_prefix,
    }

    action.update_training_job_prefix_if_not_specified(training_job_name)
    assert action.action_parameters == {
        "name": stop_training_name,
        "training_job_prefix": training_job_prefix,
    }


def test_email_action(email_name, email_address):
    """
    Validate that creating a `Email` action object is a valid action object, that it is serialized correctly into
    an action string, and that the resulting action parameters are correct.
    """
    action = Email(email_address)
    assert is_valid_action_object(action)
    assert action.action_parameters == {"name": email_name, "endpoint": email_address}


def test_sms_action(sms_name, phone_number):
    """
    Validate that creating a `SMS` action object is a valid action object, that it is serialized correctly into
    an action string, and that the resulting action parameters are correct.
    """
    action = SMS(phone_number)
    assert is_valid_action_object(action)
    assert action.action_parameters == {"name": sms_name, "endpoint": phone_number}


def test_action_list(
    stop_training_name, email_name, sms_name, training_job_name, email_address, phone_number
):
    """
    Validate that creating a `ActionList` action object (with `StopTraining`, `Email` and `SMS` actions)  is a valid
    action object, that it is serialized correctly into an action string, and that the resulting action parameters are
    correct.

    Also validate that creating this object without a custom training job prefix for the `StopTraining` action allows
    this prefix to be manually updated later on by simply calling the update function defined in the ActionList class
    (this mimics the exact behavior in sagemaker SDK, where the actual training job name is used as the prefix if
    the user did not specify a custom training job prefix).
    """
    actions = ActionList(StopTraining(), Email(email_address), SMS(phone_number))
    assert is_valid_action_object(actions)
    action_parameters = [action.action_parameters for action in actions.actions]
    assert action_parameters == [
        {"name": stop_training_name},
        {"name": email_name, "endpoint": email_address},
        {"name": sms_name, "endpoint": phone_number},
    ]

    actions.update_training_job_prefix_if_not_specified(training_job_name)
    assert action_parameters == [
        {"name": stop_training_name, "training_job_prefix": training_job_name},
        {"name": email_name, "endpoint": email_address},
        {"name": sms_name, "endpoint": phone_number},
    ]


def test_action_validation():
    """
    Validate that bad input for actions triggers an AssertionError.

    Also verify that the `is_valid_action_object` returns `False` for any input that isn't an `Action` or `ActionList`.
    This is important, as the sagemaker SDK uses this function to validate actions input.
    """
    with pytest.raises(ValueError):
        StopTraining("bad_training_job_prefix")

    with pytest.raises(ValueError):
        Email("bad.email.com")

    with pytest.raises(ValueError):
        SMS("1234")

    with pytest.raises(TypeError):
        ActionList(StopTraining(), "bad_action")

    assert not is_valid_action_object("bad_action")
