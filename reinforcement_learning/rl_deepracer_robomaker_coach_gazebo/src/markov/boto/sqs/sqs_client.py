"""This module implement sqs client"""

import logging

import botocore
from markov.boto.constants import BotoClientNames
from markov.boto.deepracer_boto_client import DeepRacerBotoClient
from markov.boto.sqs.constants import StatusIndicator
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SQS_DELETE_MESSAGE_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger

LOG = Logger(__name__, logging.INFO).get_logger()


class SQSClient(DeepRacerBotoClient):
    """
    Connects to a FIFO SQS, retrieves messages in a batch and then deletes them
    """

    name = BotoClientNames.SQS.value

    def __init__(
        self,
        queue_url,
        region_name="us-east-1",
        max_num_of_msg=1,
        wait_time_sec=5,
        max_retry_attempts=5,
        backoff_time_sec=1.0,
        session=None,
    ):
        """Initialize a sqs client with default exponital backoff retries.

        Args:
            queue_url (str): the queue url to receive message from.
            region_name (str, optional): the region name we want to create the client in.
                                         Defaults to "us-east-1".
            max_num_of_msg (int, optional): the max number of message we want to receive from the sqs queue.
                                            Defaults to 1.
            wait_time_sec (int, optional): The wait time in seconds we want to poll from sqs.
                                           Defaults to 5.
            max_retry_attempts (int, optional): The maxiumum retry attemps if something failed.
                                                Defaults to 5.
            backoff_time_sec (float, optional): The exponitial backoff time in seconds.
                                                Defaults to 1.0.
            session (boto3.Session): An alternative session to use.
                                     Defaults to None.

        """
        super(SQSClient, self).__init__(
            region_name=region_name,
            max_retry_attempts=max_retry_attempts,
            backoff_time_sec=backoff_time_sec,
            boto_client_name=self.name,
            session=session,
        )
        self._queue_url = queue_url
        self._max_num_of_msg = max_num_of_msg
        self._wait_time_sec = wait_time_sec

    def get_messages(self):
        """Fetches the SQS messages.

        Returns:
            list(str): Strips out the Body section of each message and returns all of them in a list.
            (int) 0: if no message was received
            (int) 1: if client error happened
            (int) 2: if system error happened
        """
        try:
            messages = self.get_client().receive_message(
                QueueUrl=self._queue_url,
                AttributeNames=["SentTimestamp"],
                MessageAttributeNames=["All"],
                MaxNumberOfMessages=self._max_num_of_msg,
                WaitTimeSeconds=self._wait_time_sec,
            )
            if messages.get("Messages"):
                payload = []
                entries = []
                for message in messages.get("Messages"):
                    payload.append(message["Body"])
                    entries.append(
                        {"Id": message["MessageId"], "ReceiptHandle": message["ReceiptHandle"]}
                    )
                if self.delete_messages(entries):
                    return StatusIndicator.CLIENT_ERROR.value
                LOG.info("[sqs] Received payload %s", payload)
                return payload
        except botocore.exceptions.ClientError as ex:
            LOG.error(
                "[sqs] ClientError: Unable to receive message from sqs queue %s: %s.",
                self._queue_url,
                ex,
            )
            return StatusIndicator.CLIENT_ERROR.value
        except Exception as ex:
            LOG.error(
                "[sqs] SystemError: Unable to receive message from sqs queue %s: %s.",
                self._queue_url,
                ex,
            )
            return StatusIndicator.SYSTEM_ERROR.value
        return StatusIndicator.SUCCESS.value

    def delete_messages(self, entries):
        """Deletes a group of messages from the SQS

        Args:
            entries ([dict]): A list of the messages dict to be added.
                              Each entry defines the message to be deleted by defining
                              the Id and the Receipt Handler from the original message.
        Returns:
            None:  if all messages were deleted successfully.
            [str]: list of str with details about the messages that failed to be deleted.
        """
        try:
            resp = self.exp_backoff(
                action_method=self.get_client().delete_message_batch,
                QueueUrl=self._queue_url,
                Entries=entries,
            )
            if len(resp["Successful"]) != len(entries):
                LOG.error(
                    "[sqs] ClientError: Unable to delete all read message from sqs queue %s: %s.",
                    self._queue_url,
                    resp["Failed"],
                )
                return resp["Failed"]
        except botocore.exceptions.ClientError as ex:
            LOG.error(
                "[sqs] ClientError: Unable to delete all read message from sqs queue %s: %s.",
                self._queue_url,
                ex,
            )
            # something went really wrong with the sqs queue, even though it's a client error
            # we still throw 500 because the sqs queue is likely to be supplied and managed by
            # deepracer team
            log_and_exit(
                "Exceptions in deleting message \
                         from sqs queue: {}, {}".format(
                    self._queue_url, ex
                ),
                SIMAPP_SQS_DELETE_MESSAGE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        except Exception as ex:
            LOG.error(
                "[sqs] SystemError: Unable to delete all read message from sqs queue %s: %s.",
                self._queue_url,
                ex,
            )
            # something went really wrong with the sqs queue...
            log_and_exit(
                "Exceptions in deleting message \
                         from sqs queue: {}".format(
                    self._queue_url, ex
                ),
                SIMAPP_SQS_DELETE_MESSAGE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
