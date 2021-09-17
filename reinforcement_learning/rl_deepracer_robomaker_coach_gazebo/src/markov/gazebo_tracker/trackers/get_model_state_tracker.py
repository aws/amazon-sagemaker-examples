import copy
import threading

import markov.gazebo_tracker.constants as consts
import rospy
from deepracer_msgs.srv import GetModelStates
from gazebo_msgs.srv import GetModelStateResponse
from markov.gazebo_tracker.abs_tracker import AbstractTracker
from markov.log_handler.deepracer_exceptions import GenericRolloutException
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.track_geom.constants import GET_MODEL_STATES


class GetModelStateTracker(AbstractTracker):
    """
    GetModelState Tracker class
    """

    _instance_ = None

    @staticmethod
    def get_instance():
        """Method for getting a reference to the GetModelState Tracker object"""
        if GetModelStateTracker._instance_ is None:
            GetModelStateTracker()
        return GetModelStateTracker._instance_

    def __init__(self):
        if GetModelStateTracker._instance_ is not None:
            raise GenericRolloutException("Attempting to construct multiple GetModelState Tracker")

        self.lock = threading.RLock()
        self.model_map = {}
        self.model_names = []
        self.relative_entity_names = []

        rospy.wait_for_service(GET_MODEL_STATES)
        self._get_model_states = ServiceProxyWrapper(GET_MODEL_STATES, GetModelStates)

        GetModelStateTracker._instance_ = self
        super(GetModelStateTracker, self).__init__(priority=consts.TrackerPriority.HIGH)

    def get_model_state(self, model_name, relative_entity_name, blocking=False, auto_sync=True):
        """
        Return model state of given model name based on given relative entity

        Args:
            model_name (str): name of the model
            relative_entity_name (str): relative entity name
            blocking (bool): flag to block or not
            auto_sync (bool): flag whether to automatically synchronize or not.
                              - Ignored if (model_name, relative_entity_name) pair is already using auto_sync

        Returns:
            response msg (gazebo_msgs::GetModelStateResponse)
        """
        msg = GetModelStateResponse()
        msg.success = True
        key = (model_name, relative_entity_name)
        with self.lock:
            if blocking or key not in self.model_map:
                res = self._get_model_states([key[0]], [key[1]])
                if res.success and res.status[0]:
                    model_state = res.model_states[0]
                    msg.pose = model_state.pose
                    msg.twist = model_state.twist
                    if auto_sync or key in self.model_map:
                        if key not in self.model_map:
                            self.model_names.append(model_name)
                            self.relative_entity_names.append(relative_entity_name)
                        self.model_map[key] = copy.deepcopy(model_state)
                else:
                    msg.success = False
                    msg.status_message = res.messages[0] if res.success else res.status_message
            else:
                model_state = copy.deepcopy(self.model_map[key])
                msg.pose = model_state.pose
                msg.twist = model_state.twist
        return msg

    def update_tracker(self, delta_time, sim_time):
        """
        Update model_states of the models that this tracker is tracking

        Args:
            delta_time (float): delta time
            sim_time (Clock): simulation time
        """
        if self.model_names:
            with self.lock:
                res = self._get_model_states(self.model_names, self.relative_entity_names)
                if res.success:
                    for model_state, status in zip(res.model_states, res.status):
                        if status:
                            model_name = model_state.model_name
                            relative_entity_name = model_state.reference_frame
                            key = (model_name, relative_entity_name)
                            self.model_map[key] = model_state
