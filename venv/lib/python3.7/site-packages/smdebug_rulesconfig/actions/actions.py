from typing import Optional, Union

from smdebug_rulesconfig.actions.utils import (
    validate_training_job_prefix,
    validate_email_address,
    validate_phone_number,
    validate_action_str,
)


class Action(object):
    def __init__(self, **action_parameters: str):
        """
        Base class for action, which is to be invoked when a rule fires.. Offers `serialize` function to convert action
        parameters to a string dictionary. This class is not meant to be initialized directly. Accepts dictionary of
        action parameters and drops keys whose values are `None`.

        :param action_parameters: Dictionary of action parameters.
        """
        action_parameters["name"] = self.__class__.__name__.lower()
        self.action_parameters = {
            key: value for key, value in action_parameters.items() if value is not None
        }
        validate_action_str(
            self.serialize(), self.action_parameters
        )  # sanity check, not expected to error!

    def serialize(self):
        """
        Serialize the action parameters as a string dictionary.

        :return: Action parameters serialized as a string dictionary.
        """
        return (
            "{"
            + ", ".join(
                [f'\\"{key}\\": \\"{value}\\"' for key, value in self.action_parameters.items()]
            )
            + "}"
        )


class ActionList(object):
    def __init__(self, *actions: Action):
        """
        Higher level object to maintain a list of actions to be invoked when a rule is fired. Offers higher level
        `serialize` function to handle serialization of actions as a string list of dictionaries.

        :param actions: List of actions.
        """
        if not all(isinstance(action, Action) for action in actions):
            raise TypeError("actions must be list of Action objects!")

        self.actions = actions

    def update_training_job_prefix_if_not_specified(self, training_job_name: str):
        """
        For any StopTraining actions in the action list, update the training job prefix to be the training job name if
        the user has not already specified a custom training job prefix. This is meant to be called via the sagemaker
        SDK when `estimator.fit` is called by the user. Validation is purposely excluded here so that any failures in
        validation of the training job name are intentionally caught in the sagemaker SDK and not here.

        :param training_job_name: Name of the training job, passed in when `estimator.fit` is called.
        """
        for action in self.actions:
            if isinstance(action, StopTraining):
                action.update_training_job_prefix_if_not_specified(training_job_name)

    def serialize(self):
        return "[" + ", ".join([action.serialize() for action in self.actions]) + "]"


class StopTraining(Action):
    def __init__(self, training_job_prefix: Optional[str] = None):
        """
        Action for stopping the training job when a rule is fired. Note that a policy must be created in the AWS
        account to allow the sagemaker role to stop the training job:

        ```
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "VisualEditor0",
                    "Effect": "Allow",
                    "Action": "sagemaker:StopTrainingJob",
                    "Resource": "arn:aws:sagemaker:*:<account_id>:training-job/*"
                }
            ]
        }
        ```

        :param training_job_prefix: The prefix of the training job to stop if the rule is fired. This must only refer
            to one active training job, otherwise no training job will be stopped.
        """
        self.use_default_training_job_prefix = True
        if training_job_prefix is not None:
            validate_training_job_prefix("training_job_prefix", training_job_prefix)
            self.use_default_training_job_prefix = False
        super(StopTraining, self).__init__(training_job_prefix=training_job_prefix)

    def update_training_job_prefix_if_not_specified(self, training_job_name: str):
        """
        Update the training job prefix to be the training job name if the user has not already specified a custom
        training job prefix. This is only meant to be called via the sagemaker SDK when `estimator.fit` is called by the
        user. Validation is purposely excluded here so that any failures in validation of the training job name are
        intentionally caught in the sagemaker SDK and not here.

        :param training_job_name: Name of the training job, passed in when `estimator.fit` is called.
        """
        if self.use_default_training_job_prefix:
            self.action_parameters["training_job_prefix"] = training_job_name


class Email(Action):
    def __init__(self, email_address: str):
        """
        Action for sending an email to the provided email address when the rule is fired. Note that a policy must be
        created in the AWS account to allow the sagemaker role to send an email to the user:

        ```
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "VisualEditor0",
                    "Effect": "Allow",
                    "Action": [
                        "sns:Publish",
                        "sns:CreateTopic",
                        "sns:Subscribe"
                    ],
                    "Resource": "arn:aws:sns:*:<account-id>:SMDebugRules"
                }
            ]
        }
        ```

        :param email_address: Email address to send the email notification to.
        """
        validate_email_address("email_address", email_address)
        super(Email, self).__init__(endpoint=email_address)


class SMS(Action):
    def __init__(self, phone_number: str):
        """
        Action for sending an SMS to the provided phone number when the rule is fired. Note that a policy must be
        created in the AWS account to allow the sagemaker role to send an SMS to the user:

        ```
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "VisualEditor0",
                    "Effect": "Allow",
                    "Action": [
                        "sns:Publish",
                        "sns:CreateTopic",
                        "sns:Subscribe"
                    ],
                    "Resource": "arn:aws:sns:*:<account-id>:SMDebugRules"
                }
            ]
        }
        ```

        :param phone_number: Valid phone number that follows the the E.164 format. See
            https://docs.aws.amazon.com/sns/latest/dg/sms_publish-to-phone.html for more info.
        """
        validate_phone_number("phone_number", phone_number)
        super(SMS, self).__init__(endpoint=phone_number)


def is_valid_action_object(actions: Union[Action, ActionList]):
    """
    Helper function to be used by the sagemaker SDK to determine whether the provided object is a valid action object
    or not (must be of type `Action` or `ActionList`.

    :param actions: actions object specified by the user when calling `Rule.sagemaker` in the sagemaker SDK.
    :return: Boolean for whether the provided actions object is valid or not.
    """
    return isinstance(actions, Action) or isinstance(actions, ActionList)
