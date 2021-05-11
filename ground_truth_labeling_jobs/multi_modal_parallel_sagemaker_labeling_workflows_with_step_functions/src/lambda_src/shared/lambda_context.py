import boto3


def get_account_id(context):
    """
    Retrieves the AWS account ID from a lambda context object.

    :param context: AWS Lambda context
    :type context: object
    :return: AWS account ID
    :rtype: str
    """
    # http://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html
    arn = context.invoked_function_arn
    return arn.split(":")[4]


def get_region(context):
    """
    Retrieves the AWS region the lambda is executing in.

    :param context: AWS Lambda context
    :type context: object
    :return: AWS region
    :rtype: str
    """
    # http://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html
    arn = context.invoked_function_arn
    return arn.split(":")[3]


def get_token(context):
    """
    Retrieves the token the lambda was assigned when it was invoked.

    :param context: AWS Lambda context
    :type context: object
    :return: Lambda token, usually a UUID
    :rtype: str
    """

    # If that fails, fall back to the requestID
    # http://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html
    return context.aws_request_id


def get_aws_account_id_from_arn(lambda_arn):
    """
    retrieves and return aws account id from an arn

    :param lambda_arn: arn of a calling lambda
    :type lambda_arn: string
    :returns: aws account id
    :rtype: string
    """
    return lambda_arn.split(":")[4]


def get_aws_region(service_arn):
    """
    returns AWS Region from a service arn

    :param service_arn: ARN of a service
    :type service_arn: string
    :returns: AWS Account Region
    :rtype: string
    """
    aws_region = service_arn.split(":")[3]
    return aws_region


# function to create a client with aws for a specific service and region
def get_boto_client(service, lambda_arn):
    """
    returns boto3 client for specified service

    :param service: AWS service
    :type service: string
    :param lambda_arn: ARN of a lambda function
    :type lambda_arn: string
    :param: access_key: user access key
    :type: string
    :param: secret_key: secret key
    :type: string

    :returns: returns dictionary which contains URL, bucket, key
    """
    client = boto3.client(service, get_aws_region(lambda_arn))
    return client
