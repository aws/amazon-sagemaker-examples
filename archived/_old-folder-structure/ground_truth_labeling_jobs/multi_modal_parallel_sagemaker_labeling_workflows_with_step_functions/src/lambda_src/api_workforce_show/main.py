"""
Lambda responsible for handling API requests to show the list of batches, or to
show a particular batch.
"""

import json
import os

import boto3
from shared.constants import SMGTJobCategory
from shared.log import log_request_and_context, logger


def get_member_definition_info(cognito, member_definition):
    """Builds a map of information about a given cognito user pool + group"""
    if "CognitoMemberDefinition" not in member_definition:
        logger.warning("Unknown member definition, ignoring")
        return {}

    member = member_definition["CognitoMemberDefinition"]

    member_info = {
        "userPool": member["UserPool"],
        "userGroup": member["UserGroup"],
        "clientId": member["ClientId"],
    }

    users = []
    next_token = None
    while True:
        cognito_request = {
            "UserPoolId": member_info["userPool"],
            "GroupName": member_info["userGroup"],
            "Limit": 60,
        }
        if next_token is not None:
            cognito_request["NextToken"] = next_token

        logger.info("Calling cognito list users in group with request: %s", cognito_request)

        response = cognito.list_users_in_group(**cognito_request)

        if "Users" not in response:
            logger.warning("missing users key in response, just returning ids")
            return member_info

        users += response["Users"]

        if "NextToken" not in response:
            break

        next_token = response["NextToken"]

    user_info_list = []
    for user in users:
        # TODO: Can add additional user attributes here.
        user_info_list.append(
            {
                "username": user["Username"],
                "enabled": user["Enabled"],
                "userStatus": user["UserStatus"],
                "userCreateDate": user["UserCreateDate"],
                "userLastModifiedDate": user["UserLastModifiedDate"],
            }
        )

    member_info["members"] = user_info_list

    return member_info


def handle_request():
    """Generates information about deployed workforces"""

    sagemaker = boto3.client("sagemaker")

    workteam_name_by_category = {
        SMGTJobCategory.PRIMARY: os.getenv("FIRST_LEVEL_WORKTEAM_NAME"),
        SMGTJobCategory.SECONDARY: os.getenv("SECOND_LEVEL_WORKTEAM_NAME"),
    }

    sagemaker = boto3.client("sagemaker")
    cognito = boto3.client("cognito-idp")

    workforce_info = []
    for category, workteam_name in workteam_name_by_category.items():
        workteam_info = {
            "category": category,
            "workteamName": workteam_name,
        }

        response = sagemaker.describe_workteam(WorkteamName=workteam_name)
        logger.info("Sagemaker describe workteam %s response: %s", workteam_name, response)

        if not "Workteam" in response or not "MemberDefinitions" in response["Workteam"]:
            # We'll catch, log and handle at a higher level, let this bubble up.
            raise Exception("Couldn't get member definitions from work team")

        member_infos = []

        member_definitions = response["Workteam"]["MemberDefinitions"]
        for member_definition in member_definitions:
            member_info = get_member_definition_info(cognito, member_definition)
            member_infos.append(member_info)

        workteam_info["memberDefinitions"] = member_infos

        workforce_info.append(workteam_info)

    return workforce_info


def lambda_handler(event, context):
    """Lambda function that responds shows active batch information.

    Parameters
    ----------
    event: dict, required API gateway request with an input SQS arn, output SQS arn
    context: object, required Lambda Context runtime methods and attributes
    Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    Lambda Output Format: dict
    Return doc:
    https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """
    log_request_and_context(event, context)

    try:
        response = handle_request()
    # Allow because we want to control status code and message if request has any failure.
    # pylint: disable=broad-except
    except Exception as err:
        logger.error("Failed to handle request for workforce: %s", err)
        return {
            "statusCode": 500,
            "body": "Error: failed to handle request.",
        }

    response = {
        "statusCode": 200,
        "body": json.dumps(response, default=str),
        "isBase64Encoded": False,
    }
    return response
