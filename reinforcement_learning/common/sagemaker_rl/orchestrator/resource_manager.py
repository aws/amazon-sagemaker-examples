import json
import logging
import time

import boto3
import sagemaker
from botocore.exceptions import ClientError
from sagemaker.local.local_session import LocalSession
from sagemaker.predictor import RealTimePredictor

from orchestrator.exceptions.ddb_client_exceptions import RecordAlreadyExistsException

from boto3.dynamodb.conditions import Key

from orchestrator.clients.ddb.experiment_db_client import ExperimentDbClient
from orchestrator.clients.ddb.join_db_client import JoinDbClient
from orchestrator.clients.ddb.model_db_client import ModelDbClient

logger = logging.getLogger(__name__)

class ResourceManager(object):
    """A resource manager entity to manage computing resource creation
    and cleanup for the experiment.
    """

    def __init__(
        self,
        resource_config,
        boto_session=None
    ):
        """Initialize a resource manager entity given a resource config
        
        Args:
            resource_config (dict): A dictionary containing configuration
                of the computing resource
            boto_session (boto3.session.Session): A session stores configuration
                state and allows you to create service clients and resources.
        """
        if boto_session is None:
            boto_session = boto3.Session()
        self.boto_session = boto_session

        # Initialize resource clients
        self.cf_client = boto3.client("cloudformation")
        self.firehose_client = self.boto_session.client("firehose")
        self.exp_db_client = None
        self.model_db_client = None
        self.join_db_client = None

        # load config
        self._resource_config = resource_config
        self.shared_resource_stack_name = self._resource_config.get("shared_resource").get("resources_cf_stack_name")

        self.hosting_fleet_config = self._resource_config.get("private_resource").get("hosting_fleet")
        self.training_fleet_config = self._resource_config.get("private_resource").get("training_fleet")
        self.evaluation_fleet_config = self._resource_config.get("private_resource").get("evaluation_fleet")

    @property
    def firehose_bucket(self):
        if hasattr(self, 'firehose_s3_bucket_name'):
            return self.firehose_s3_bucket_name
        account = self.boto_session.client("sts").get_caller_identity()["Account"]
        region = self.boto_session.region_name
        # Use sagemaker bucket to store firehose data
        firehose_s3_bucket_name = "{}-{}-{}".format("sagemaker", region, account)
        self.firehose_s3_bucket_name = firehose_s3_bucket_name
        return firehose_s3_bucket_name

    def create_shared_resource_if_not_exist(self):
        """Create shared resource across experiments, including
        experiment ddb table, joining job ddb table, model ddb table
        and IAM role to grant relevant resource permission
        """
        if self._usable_shared_cf_stack_exists():
            logger.info("Using Resources in CloudFormation stack named: {} " \
                "for Shared Resources.".format(self.shared_resource_stack_name))
        else:
            logger.info("Creating a new CloudFormation stack for Shared Resources. " \
                "You can always reuse this StackName in your other experiments")
            self._create_new_cloudformation_stack()

        # use Output Resources Names from CloudFromation stack
        self.exp_db_table_name = self._get_cf_output_by_key('ExperimentDbTableName')
        self.join_db_table_name = self._get_cf_output_by_key('JoinDbTableName')
        self.model_db_table_name = self._get_cf_output_by_key("ModelDbTableName")
        self.iam_role_arn = self._get_cf_output_by_key('IAMRoleArn')
        
        # initialize DynamoDb clients!
        experiment_db_session = self.boto_session.resource('dynamodb').Table(self.exp_db_table_name)
        self.exp_db_client = ExperimentDbClient(experiment_db_session)

        join_db_session = self.boto_session.resource('dynamodb').Table(self.join_db_table_name)
        self.join_db_client = JoinDbClient(join_db_session)

        model_db_session = self.boto_session.resource('dynamodb').Table(self.model_db_table_name)
        self.model_db_client = ModelDbClient(model_db_session)

    def _usable_shared_cf_stack_exists(self):
        """Check if the shared cf stack exist and is usable
        
        Returns:
            bool: Whether the shared cf stack is usable
        """
        # we can remove this logic, and have checks only on CF stack exists,
        # CF stack in one of [CREATE|UPDATE|ROLLBACK]_COMPLETE state
        try:
            stack_name = self.shared_resource_stack_name
            response = self.cf_client.describe_stacks(
                StackName=stack_name)["Stacks"]
            if len(response) == 0:
                return False
        except Exception as e:
            if "UnauthorizedOperation" in str(e):
                raise Exception("You are unauthorized to describe a CloudFormation Stack. Please update your Role with "
                                " appropriate permissions.")
            elif "ValidationError" in str(e):
                # stack doesn't exists
                return False
            else:
                raise e
    
        stack_details = response[0]
        stack_status = stack_details['StackStatus']
        if stack_status in ['UPDATE_COMPLETE', 'CREATE_COMPLETE']:
            return True
        elif stack_status in ["DELETE_COMPLETE"]:
            return False
        elif stack_status in ["ROLLBACK_COMPLETE"]:
            logger.error(f"Stack with name {stack_name} is in {stack_status} state! Please delete/ stabilize/ or "
                        "or update Config.yaml to create a new stack")
            raise Exception(f"A Cloudformation Stack with name {stack_name}, already exists in {stack_status} State. "
                            f"Please debug/ or delete the stack here: {self._get_cf_stack_events_link()}"
            )
        elif "FAILED" in stack_status:
            logger.error(f"Stack with name {stack_name} in {stack_status} state! Please delete the stack"
                         " or update Config.yaml to create a new stack")
            raise Exception(f"A Cloudformation Stack with name {stack_name}, already exists in {stack_status} State. "
                            f"Please debug/ or delete the stack here: {self._get_cf_stack_events_link()}"
            )
        elif "DELETE" in stack_status:
            # already checked DELETE_COMPLETE above
            logger.error("Stack with name {} is in {} state! Cannot continue further!" \
                " Please wait for the delete to complete".format(stack_name, stack_status))
            raise Exception(f"A Cloudformation Stack with name {stack_name}, already exists in {stack_status} State. "
                            f"Please retry after the stack gets Deleted/or debug the stack here: {self._get_cf_stack_events_link()}"
            )
        elif "CREATE" in stack_status:
            # one of the create statuses!
            logger.info("Stack with name {} exists in {} state".format(stack_name, stack_status))
            logger.warn("Waiting for stack to get to CREATE_COMPLETE state....")
            self._wait_for_cf_stack_create_to_complete()
            return True
        else:
            # assume stack in modifying. wait for it to goto 
            logger.info("Stack in {} state. Waiting for it's to end in successful state...".format(stack_status))
            self._wait_for_cf_stack_update_to_complete()
            return True


    def _create_new_cloudformation_stack(self):
        """Create a new cloudformation stack
        
        Returns:
            bool: whether successfully create a new cloudformation stack
        """
        try:
            cf_stack_name = self.shared_resource_stack_name
            parameters = [
                {
                    "ParameterKey": "IAMRoleName",
                    "ParameterValue": self._get_iam_role_property('role_name', 'role_for_cl'),
                    "UsePreviousValue": True,
                    "ResolvedValue": "string"
                },
            ]
            parameters.extend(self._get_cloudformation_parameters_for_db())
            logger.info(json.dumps(parameters, indent=4))

            self.cf_client.create_stack(
                StackName=cf_stack_name,
                TemplateBody=self._parse_template(),
                Parameters=parameters,
                Capabilities=[
                    'CAPABILITY_NAMED_IAM'
                ]
            )
            logger.info("Creating CloudFormation Stack for shared resource!")
            self._wait_for_cf_stack_create_to_complete()
            return True
        except Exception as  e:
            if "UnauthorizedOperation" in str(e):
                raise Exception("You are unauthorized to create a CloudFormation Stack. Please update your Role with "
                                " appropriate permissions.")
            elif "AlreadyExists" in str(e):
                # it came here it means it must be in one for "CREATING states"
                logger.warn(f"A stack with name {cf_stack_name} already exists. Reusing the stack" \
                             " resources for this experiment")
                self._wait_for_cf_stack_create_to_complete()
                return False
            raise(e)

    def _get_cf_stack_events_link(self):
        """Get events link for the given shared cf stack
        
        Returns:
            str: events link for the cf stack
        """
        region = self.boto_session.region_name
        # update for non-commercial system
        return f"https://{region}.console.aws.amazon.com/cloudformation/home?region={region}#/stacks/events?stackId={self.shared_resource_stack_name}"

    def _wait_for_cf_stack_create_to_complete(self):
        """Wait until the cf stack creation complete
        """
        cf_waiter = self.cf_client.get_waiter('stack_create_complete')
        logger.info("Waiting for stack to get to CREATE_COMPLETE state....")
        try:
            cf_waiter.wait(
                StackName=self.shared_resource_stack_name,
                WaiterConfig={
                    'Delay': 10,
                    'MaxAttempts': 60
                }
            )
        except Exception as e:
            logger.error(e)
            logger.error("Failed to Create Stack with name {} ".format(self.shared_resource_stack_name))
            raise Exception(f"Failed to Create Shared Resource Stack. "
                            f"Please debug the stack here: {self._get_cf_stack_events_link()}"
            )

    def _wait_for_cf_stack_update_to_complete(self):
        """Wait until the cf stack update complete
        """
        cf_waiter = self.cf_client.get_waiter('stack_update_complete')
        logger.info("Waiting for stack to get to Successful Update state....")
        try:
            cf_waiter.wait(
                StackName=self.shared_resource_stack_name,
                WaiterConfig={
                    'Delay': 10,
                    'MaxAttempts': 6
                }
            )
        except Exception as e:
            logger.error(e)
            logger.error("Failed to use Stack with name {} ".format(self.shared_resource_stack_name))
            raise Exception(f"The provided CloudFormation Stack for Shared Resource is unstable. "
                            f"Please debug the stack here: {self._get_cf_stack_events_link()}"
            )

    def _parse_template(self):
        """Parse Yaml file for cloudformation

        Return:
            str: content in the template file
        """
        with open("./common/sagemaker_rl/orchestrator/cloudformation.yaml") as template_fileobj:
            template_data = template_fileobj.read()
        self.cf_client.validate_template(TemplateBody=template_data)
        return template_data

    def _get_cloudformation_parameters_for_db(self):
        """Return config values for each ddb table

        Returns:
            list: A list json blobs containing config values
                for each ddb table
        """
        json_parameter_list = []
        cf_parameter_prefixes = ["ExperimentDb", "ModelDb", "JoinDb"]

        for parameter_prefix in cf_parameter_prefixes:
            json_params = [
                {
                    "ParameterKey": parameter_prefix + "Name",
                    "ParameterValue": self._get_resource_property(parameter_prefix, "table_name"),
                    "UsePreviousValue": True,
                    "ResolvedValue": "string"
                }, 
                {
                    "ParameterKey": parameter_prefix + "RCU",
                    "ParameterValue": self._get_resource_property(parameter_prefix, "rcu", '5'),
                    "UsePreviousValue": True,
                    "ResolvedValue": "string"
                }, 
                {
                    "ParameterKey": parameter_prefix + "WCU",
                    "ParameterValue": self._get_resource_property(parameter_prefix, "wcu", '5'),
                    "UsePreviousValue": True,
                    "ResolvedValue": "string"
                }
            ]
            json_parameter_list.extend(json_params)
        return json_parameter_list

    def _get_resource_property(self, resource_name, property_name, default_value=None):
        """Get property value of given resource
        
        Args:
            resource_name (str): Name of the resource 
            property_name (str): Name of the property
            default_value (str): Default value of the property
        
        Returns:
            str: Property value of the resource
        """
        if resource_name == "ExperimentDb":
            return self._get_experiment_db_property(property_name, default_value)
        elif resource_name == "ModelDb":
            return self._get_model_db_property(property_name, default_value)
        elif resource_name == "JoinDb":
            return self._get_join_db_property(property_name, default_value)
        elif resource_name == "IAMRole":
            return self._get_iam_role_property(property_name, default_value)
        else:
            return None

    def _get_experiment_db_property(self, property_name, default_value=None):
        """Return property value of experiment table
        Args:
            property_name (str): name of property
            default_value (): default value of the property
        
        Returns:
            value of the property
        """
        experiment_db_config = self._resource_config.get("shared_resource").get("experiment_db")
        return experiment_db_config.get(property_name, default_value)
    
    def _get_model_db_property(self, property_name, default_value=None):
        """Return property value of model table
        Args:
            property_name (str): name of property
            default_value (): default value of the property
        
        Returns:
            value of the property
        """
        model_db_config = self._resource_config.get("shared_resource").get("model_db")
        return model_db_config.get(property_name, default_value)

    def _get_join_db_property(self, property_name,default_value=None):
        """Return property value of join table
        Args:
            property_name (str): name of property
            default_value (): default value of the property
        
        Returns:
            value of the property
        """        
        join_db_config = self._resource_config.get("shared_resource").get("join_db")
        return join_db_config.get(property_name, default_value)
    
    def _get_iam_role_property(self, property_name, default_value=None):
        """Return property value of iam role
        Args:
            property_name (str): name of property
            default_value (): default value of the property
        
        Returns:
            value of the property
        """
        iam_role_config = self._resource_config.get("shared_resource").get("iam_role")
        return iam_role_config.get(property_name, default_value)

    def _get_cf_output_by_key(self, output_key):
        """Return cf output value of given output key
        
        Args:
            output_key (str): key of a specific output
        
        Returns:
            str: value of the output key
        """
        stack_json = self.cf_client.describe_stacks(
            StackName=self.shared_resource_stack_name
        )["Stacks"][0]
    
        # validate stack has been successfully updater
        if stack_json["StackStatus"] not in \
                ["CREATE_COMPLETE", "UPDATE_COMPLETE", 
                "ROLLBACK_COMPLETE", "UPDATE_ROLLBACK_COMPLETE"]:
            logger.error("Looks like Resource CF Stack is in {} state. " \
                "Cannot continue forward. ".format(stack_json["StackStatus"]))
            raise Exception("Please wait while the Shared Resources Stack gets into a usable state." \
                "Currently in state {}!".format(stack_json["StackStatus"]))

        stack_outputs = stack_json["Outputs"]
        for stack_output in stack_outputs:
            if stack_output["OutputKey"] == output_key:
                return stack_output["OutputValue"]
        raise Exception("Cannot find an Output with OutputKey {} in Shared CF Stack".format(output_key))

    def _wait_for_active_firehose(self, stream_name):
        """Wait until the firehose stream creation complete and be active
        
        Args:
            stream_name (str): stream name of the firehose
        """
        status = 'CREATING'
        timeout = 60 * 2
        while status != 'ACTIVE' and timeout >= 0:
            logger.info("Creating firehose delivery stream...")
            try:
                result = self.firehose_client.describe_delivery_stream(DeliveryStreamName=stream_name)
            except ClientError as e:
                error_code = e.response['Error']['Code']
                message = e.response['Error']['Message']
                raise RuntimeError(f"Failed to describe delivery stream '{stream_name}' "
                                   f"with error {error_code}: {message}")
            status = result['DeliveryStreamDescription']['DeliveryStreamStatus']
            time.sleep(10)
            timeout = timeout - 10
        if status == 'ACTIVE':
            logger.info(f"Successfully created delivery stream '{stream_name}'")
        else:
            raise RuntimeError(f"Failed to create delivery stream '{stream_name}'")

    def _init_firehose_from_config(self, stream_name, s3_bucket, s3_prefix,
                                   buffer_size=128, buffer_time=60):
        """Initiate a firehose stream with given config
        
        Args:
            stream_name (str): name of the firehose stream
            s3_bucket (str): s3 bucket for delivering the firehose streaming data
            s3_prefix (str): s3 prefix path for delivering the firehose data
            buffer_size (int): buffer size(MB) in firehose before pushing data
                to S3 destination
            buffer_time (int): buffer time(s) in firehose before pushing
                data to s3 destination
        """
        exist_delivery_streams = self.firehose_client.list_delivery_streams(Limit=1000)['DeliveryStreamNames']
        if stream_name in exist_delivery_streams:
            logger.warning(f"Delivery stream {stream_name} already exist. "
                           "No new delivery stream created.")
        else:
            firehose_role_arn = self.iam_role_arn
            s3_bucket_arn = f"arn:aws:s3:::{s3_bucket}"

            s3_config = {
                'BucketARN': s3_bucket_arn,
                'RoleARN': firehose_role_arn,
                'Prefix': s3_prefix.strip() + '/',
                'BufferingHints': {
                    'IntervalInSeconds': buffer_time,
                    'SizeInMBs': buffer_size
                },
            }

            try:
                self.firehose_client.create_delivery_stream(
                    DeliveryStreamName=stream_name,
                    DeliveryStreamType='DirectPut',
                    ExtendedS3DestinationConfiguration=s3_config
                )
            except ClientError as e:
                error_code = e.response['Error']['Code']
                message = e.response['Error']['Message']
                raise RuntimeError(f"Failed to create delivery stream '{stream_name}' "
                                   f"with error {error_code}: {message}")

            # check if delivery stream created
            self._wait_for_active_firehose(stream_name)

    def create_firehose_stream_if_not_exists(self, stream_name, s3_prefix):
        """Create firehose stream with given stream name
        
        Arguments:
            stream_name (str): name of the firehose stream
            s3_prefix (str): s3 prefix path for delivering the firehose data
        """
        # use sagemaker-{region}-{account_id} bucket to store data
        self.firehose_s3_bucket_name = self._create_s3_bucket_if_not_exist("sagemaker")
        self._init_firehose_from_config(stream_name, self.firehose_s3_bucket_name, s3_prefix)

    def delete_firehose_stream(self, stream_name):
        """Delete the firehose with given stream name
        
        Args:
            stream_name (str): name of the firehose stream
        """
        logger.warning(f"Deleting firehose stream '{stream_name}'...")

        try:
            self.firehose_client.delete_delivery_stream(
                DeliveryStreamName=stream_name
            )
        except ClientError as e:
            error_code = e.response['Error']['Code']
            message = e.response['Error']['Message']
            raise RuntimeError(f"Failed to delete delivery stream '{stream_name}' "
                                f"with error {error_code}: {message}")

    def _create_s3_bucket_if_not_exist(self, prefix):
        """Create s3 bucket if not exist
        
        Args:
            prefix (str): A bucket name prefix, followed by region name
                and account id 
        
        Returns:
            str: s3 bucket name
        """
        account = self.boto_session.client("sts").get_caller_identity()["Account"]
        region = self.boto_session.region_name
        s3_bucket_name = "{}-{}-{}".format(prefix, region, account)

        s3 = self.boto_session.resource("s3")
        s3_client = self.boto_session.client("s3")
        try:
            # 'us-east-1' cannot be specified because it is the default region:
            # https://github.com/boto/boto3/issues/125
            if region == "us-east-1":
                s3.create_bucket(Bucket=s3_bucket_name)
            else:
                s3.create_bucket(
                    Bucket=s3_bucket_name, CreateBucketConfiguration={"LocationConstraint": region}
                )
            logger.info("Successfully create S3 bucket '{}' for storing {} data".format(s3_bucket_name, prefix))
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            message = e.response["Error"]["Message"]

            if error_code == "BucketAlreadyOwnedByYou":
                pass
            elif (
                error_code == "OperationAborted" and "conflicting conditional operation" in message
            ):
                # If this bucket is already being concurrently created, we don't need to create it again.
                pass
            elif error_code == "TooManyBuckets":
                # Succeed if the default bucket exists
                s3.meta.client.head_bucket(Bucket=s3_bucket_name)
            else:
                raise
        
        s3_waiter = s3_client.get_waiter('bucket_exists')
        s3_waiter.wait(Bucket=s3_bucket_name)
        return s3_bucket_name


class Predictor(object):
    def __init__(self, endpoint_name, sagemaker_session=None):
        """
        Args:
            endpoint_name (str): name of the Sagemaker endpoint
            sagemaker_session (sagemaker.session.Session): Manage interactions
                with the Amazon SageMaker APIs and any other AWS services needed.
        """
        self.endpoint_name = endpoint_name
        self._realtime_predictor = RealTimePredictor(endpoint_name,
                                                     serializer=sagemaker.predictor.json_serializer,
                                                     deserializer=sagemaker.predictor.json_deserializer,
                                                     sagemaker_session=sagemaker_session)

    def get_action(self, obs=None):
        """Get prediction from the endpoint
        
        Args:
            obs (list/str): observation of the environment

        Returns:
            action: action to take from the prediction
            event_id: event id of the current prediction
            model_id: model id of the hosted model
            action_prob: action probability distribution
            sample_prob: sample probability distribution used for data split
        """
        payload = {}
        payload['request_type'] = "observation"
        payload['observation'] = obs
        response = self._realtime_predictor.predict(payload)
        action = response['action']
        action_prob = response['action_prob']
        event_id = response['event_id']
        model_id = response['model_id']
        sample_prob = response['sample_prob']
        return action, event_id, model_id, action_prob, sample_prob

    def get_hosted_model_id(self):
        """Return hostdd model id in the hosting endpoint
        
        Returns:
            str: model id of the model being hosted
        """
        payload = {}
        payload['request_type'] = "model_id"
        payload['observation'] = None
        response = self._realtime_predictor.predict(payload)
        model_id = response['model_id']

        return model_id

    def delete_endpoint(self):
        """Delete the Sagemaker endpoint
        """
        logger.warning(f"Deleting hosting endpoint '{self.endpoint_name}'...")
        self._realtime_predictor.delete_endpoint()