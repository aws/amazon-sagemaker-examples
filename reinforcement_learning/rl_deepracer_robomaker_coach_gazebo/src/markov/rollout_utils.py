"""This module should contain common utility methods for the rollout woeker that
   depend on ros, it should not be used in modules imported to the training worker
"""
import logging

import rospy
from deepracer_msgs.srv import GetLightNames, GetLightNamesRequest
from markov.agents.utils import RunPhaseSubject
from markov.common import ObserverInterface
from markov.constants import (
    ROBOMAKER_IS_PROFILER_ON,
    ROBOMAKER_PROFILER_S3_BUCKET,
    ROBOMAKER_PROFILER_S3_PREFIX,
)
from markov.domain_randomizations.constants import GazeboServiceName
from markov.domain_randomizations.randomizer_manager import RandomizerManager
from markov.domain_randomizations.visual.light_randomizer import LightRandomizer
from markov.log_handler.logger import Logger
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.utils import str2bool
from rl_coach.core_types import RunPhase
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

LOG = Logger(__name__, logging.INFO).get_logger()
# Mapping of the phase to the text to display
PHASE_TO_MSG_MAP = {RunPhase.HEATUP: 0, RunPhase.TRAIN: 1, RunPhase.TEST: 2, RunPhase.UNDEFINED: 3}
# Messages to display on the video
HEAT = "Heatup"
TRAIN = "Training"
EVAL = "Evaluating"
IDLE = "Idle"
UNKWN = "Unknown"
# Allowed label transitions for going from one run phase to another
LABEL_TRANSITION = [
    [HEAT, TRAIN, EVAL, IDLE],
    [UNKWN, TRAIN, EVAL, IDLE],
    [UNKWN, TRAIN, EVAL, EVAL],
    [HEAT, TRAIN, EVAL, IDLE],
]


def signal_robomaker_markov_package_ready():
    """Since the ROS nodes are asynchronous, there is no guarantee that the required
    node is up and running and all the information are populated. In some cases, the node
    launched from the simulation application would have to wait for all the markov package
    to be ready.
    In one of the scenario, where the mapping of agents/bots/obstacles on the track as icon
    has to wait for all the models to spawn up. The getModelState would throw exception,
    because gazebo service might be up but the spawn model in markov package is not populated
    with all the obstacles, agents and bot. The current approach of trying to getModel and
    then see if the value present would pollute the logs. To avoid this a ROS service is
    created. This ros-service will be up only when all the Markov package is ready. This way
    the client can wait on this service and need not have to have a seperate logic.
    """
    rospy.Service("/robomaker_markov_package_ready", Empty, handle_robomaker_markov_package_ready)


def handle_robomaker_markov_package_ready():
    """This is the handler for responding to the request to check if markov robomaker package
    is up and all the required data is available.

    Returns:
        EmptyResponse: An empty response stating its ready
    """
    return EmptyResponse()


class PhaseObserver(ObserverInterface):
    """Class that gets notified when the phase changes and publishes the phase to
    a desired topic
    """

    def __init__(self, topic: str, sink: RunPhaseSubject) -> None:
        """topic - Topic for which to publish the phase"""
        self._phase_pub_ = rospy.Publisher(topic, String, queue_size=1)
        self._state_ = None
        sink.register(self)

    def update(self, data: str) -> None:
        try:
            new_state = PHASE_TO_MSG_MAP[data]
            msg = (
                LABEL_TRANSITION[self._state_][new_state]
                if self._state_
                else LABEL_TRANSITION[new_state][new_state]
            )
            self._state_ = new_state
        except KeyError:
            LOG.info("Unknown phase: %s", data)
            msg = UNKWN
            self._state_ = None
        self._phase_pub_.publish(msg)


def configure_environment_randomizer(light_name_filter=None):
    rospy.wait_for_service(GazeboServiceName.GET_LIGHT_NAMES.value)
    get_light_names = ServiceProxyWrapper(GazeboServiceName.GET_LIGHT_NAMES.value, GetLightNames)
    res = get_light_names(GetLightNamesRequest())
    for light_name in res.light_names:
        if light_name_filter and light_name not in light_name_filter:
            continue
        RandomizerManager.get_instance().add(LightRandomizer(light_name=light_name))


def get_robomaker_profiler_env():
    """Read robomaker profiler environment"""
    is_profiler_on = str2bool(rospy.get_param(ROBOMAKER_IS_PROFILER_ON, False))
    profiler_s3_bucker = rospy.get_param(ROBOMAKER_PROFILER_S3_BUCKET, None)
    profiler_s3_prefix = rospy.get_param(ROBOMAKER_PROFILER_S3_PREFIX, None)
    return is_profiler_on, profiler_s3_bucker, profiler_s3_prefix
