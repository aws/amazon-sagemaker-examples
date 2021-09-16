import threading

import markov.gazebo_tracker.constants as consts
import rospy
from deepracer_msgs.srv import SetModelStates
from gazebo_msgs.srv import SetModelStateResponse
from markov.gazebo_tracker.abs_tracker import AbstractTracker
from markov.log_handler.deepracer_exceptions import GenericRolloutException
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.track_geom.constants import SET_MODEL_STATES


class SetModelStateTracker(AbstractTracker):
    """
    SetModelState Tracker class
    """

    _instance_ = None

    @staticmethod
    def get_instance():
        """Method for getting a reference to the SetModelState Tracker object"""
        if SetModelStateTracker._instance_ is None:
            SetModelStateTracker()
        return SetModelStateTracker._instance_

    def __init__(self):
        if SetModelStateTracker._instance_ is not None:
            raise GenericRolloutException("Attempting to construct multiple SetModelState Tracker")

        self.lock = threading.RLock()
        self.model_state_map = {}

        rospy.wait_for_service(SET_MODEL_STATES)
        self.set_model_states = ServiceProxyWrapper(SET_MODEL_STATES, SetModelStates)

        SetModelStateTracker._instance_ = self
        super(SetModelStateTracker, self).__init__(priority=consts.TrackerPriority.LOW)

    def set_model_state(self, model_state, blocking=False):
        """
        Set ModelState that will be updated in next update call

        Args:
            model_state (ModelState): the model state to update
            blocking (bool): flag to block or not
        Returns:
            msg (SetModelStateResponse)
        """
        msg = SetModelStateResponse()
        msg.success = True
        with self.lock:
            if blocking:
                if model_state.model_name in self.model_state_map:
                    del self.model_state_map[model_state.model_name]
                res = self.set_model_states([model_state])
                msg.success = res.success and res.status[0]
                msg.status_message = res.messages[0] if res.success else res.status_message
            else:
                self.model_state_map[model_state.model_name] = model_state
        return msg

    def update_tracker(self, delta_time, sim_time):
        """
        Update all model states tracking to gazebo

        Args:
            delta_time (float): delta time
            sim_time (Clock): simulation time
        """
        with self.lock:
            if self.model_state_map.values():
                self.set_model_states(self.model_state_map.values())
            self.model_state_map = {}
