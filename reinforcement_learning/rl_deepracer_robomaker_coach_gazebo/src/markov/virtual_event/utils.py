"""This module prodives utils for virtual event."""
import json
import logging
from datetime import datetime

from jsonschema import validate
from jsonschema.exceptions import ValidationError
from markov.log_handler.constants import SIMAPP_EVENT_ERROR_CODE_500, SIMAPP_EVENT_SYSTEM_ERROR
from markov.log_handler.deepracer_exceptions import GenericNonFatalException
from markov.log_handler.logger import Logger
from markov.virtual_event.constants import (
    CAR_CONTROL_INPUT_SCHEMA,
    RACER_INFO_JSON_SCHEMA,
    TIME_FORMAT,
    CarControlMode,
    CarControlStatus,
    CarControlTopic,
    WebRTCCarControl,
)

LOG = Logger(__name__, logging.INFO).get_logger()


def validate_json_input(message_body, json_schema):
    """Validate the racer_info json we receive.

    Args:
        message_body (str): a json string to validate
        json_schema (dict): json schema format to conform to.

    Raises:
        GenericNonFatalException: If the message has some problem,
                                  we allow new messages to be retried
                                  without exiting the simapp.
    """
    try:
        validate(instance=json.loads(message_body), schema=json_schema)
    except ValidationError as ex:
        error_msg = "[json validation] Invalid json format: {}.".format(ex)
        # even thought it's a client error (the entity calling simapp sending us wrong input message)
        # we are calling it 500 here because:
        # 1. it's the cloud platform team sending us the message and
        # we treat the cloud and the simapp as one internal system for deepracer.
        # 2. a contract is already defined with the cloud which we should honor.
        raise GenericNonFatalException(
            error_msg=error_msg,
            error_code=SIMAPP_EVENT_ERROR_CODE_500,
            error_name=SIMAPP_EVENT_SYSTEM_ERROR,
        )
    except Exception as ex:
        error_msg = "[json validation] Something wrong when validating json format: {}.".format(ex)
        raise GenericNonFatalException(
            error_msg=error_msg,
            error_code=SIMAPP_EVENT_ERROR_CODE_500,
            error_name=SIMAPP_EVENT_SYSTEM_ERROR,
        )


def process_car_control_msg(message):
    """Process the car control msg.

    Args:
        message (str): The message to process for car control.

    Raises:
        GenericNonFatalException: If processing message has some problem,
                                  it's not fatal and we ignore it and continue.

    Returns:
        topic (str): the control topic to publish to.
        payload (dict): the actual payload for the control topic.
    """
    try:
        validate_json_input(message, CAR_CONTROL_INPUT_SCHEMA)
        msg_json = json.loads(message)
        control_type = msg_json[WebRTCCarControl.TYPE.value]
        if control_type == WebRTCCarControl.STATUS.value:
            topic = CarControlTopic.STATUS_CTRL.value
        elif control_type == WebRTCCarControl.SPEED.value:
            topic = CarControlTopic.SPEED_CTRL.value
        # log latency
        log_latency(msg_json)
        return topic, msg_json[WebRTCCarControl.PAYLOAD.value]
    except GenericNonFatalException as ex:
        raise ex
    except Exception as ex:
        error_msg = "[webrtc msg process] Exception in processing \
                    webrtc message: {}, {}".format(
            message, ex
        )
        raise GenericNonFatalException(
            error_msg=error_msg,
            error_code=SIMAPP_EVENT_ERROR_CODE_500,
            error_name=SIMAPP_EVENT_SYSTEM_ERROR,
        )


def log_latency(msg_json):
    """Log the latency for webrtc data channel for reference.

    Args:
        msg_json (dict): The input message dictionary
    """
    epoch_sec = int(msg_json[WebRTCCarControl.SENT_TIME.value]) / 1000.0
    sent_time = datetime.fromtimestamp(epoch_sec)
    receive_time = datetime.utcnow()
    difference = receive_time - sent_time
    LOG.info("[webrtc msg process] sent_time: %s", sent_time.strftime(TIME_FORMAT))
    LOG.info("[webrtc msg process] receive_time: %s", receive_time.strftime(TIME_FORMAT))
    LOG.info("[webrtc msg process] message latency %s", difference)
