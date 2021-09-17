import copy
import threading

import markov.gazebo_tracker.constants as consts
import rospy
from deepracer_msgs.srv import GetLinkStates
from gazebo_msgs.srv import GetLinkStateResponse
from markov.gazebo_tracker.abs_tracker import AbstractTracker
from markov.log_handler.deepracer_exceptions import GenericRolloutException
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.track_geom.constants import GET_LINK_STATES


class GetLinkStateTracker(AbstractTracker):
    """
    GetLinkState Tracker class
    """

    _instance_ = None

    @staticmethod
    def get_instance():
        """Method for getting a reference to the GetLinkState Tracker object"""
        if GetLinkStateTracker._instance_ is None:
            GetLinkStateTracker()
        return GetLinkStateTracker._instance_

    def __init__(self):
        if GetLinkStateTracker._instance_ is not None:
            raise GenericRolloutException("Attempting to construct multiple GetLinkState Tracker")

        self.lock = threading.RLock()
        self.link_map = {}
        self.link_names = []
        self.reference_frames = []

        rospy.wait_for_service(GET_LINK_STATES)
        self._get_link_states = ServiceProxyWrapper(GET_LINK_STATES, GetLinkStates)

        GetLinkStateTracker._instance_ = self
        super(GetLinkStateTracker, self).__init__(priority=consts.TrackerPriority.HIGH)

    def get_link_state(self, link_name, reference_frame, blocking=False, auto_sync=True):
        """
        Return link state of given link name based on given reference frame

        Args:
            link_name (str): name of the link
            reference_frame (str): reference frame
            blocking (bool): flag to block or not
            auto_sync (bool): flag whether to automatically synchronize or not.
                              - Ignored if (model_name, relative_entity_name) pair is already using auto_sync

        Returns:
            response msg (gazebo_msgs::GetLinkStateResponse)
        """
        msg = GetLinkStateResponse()
        msg.success = True
        key = (link_name, reference_frame)
        with self.lock:
            if blocking or key not in self.link_map:
                res = self._get_link_states([key[0]], [key[1]])
                if res.success and res.status[0]:
                    msg.link_state = res.link_states[0]
                    if auto_sync or key in self.link_map:
                        if key not in self.link_map:
                            self.link_names.append(link_name)
                            self.reference_frames.append(reference_frame)
                        self.link_map[key] = copy.deepcopy(msg.link_state)
                else:
                    msg.success = False
                    msg.status_message = res.messages[0] if res.success else res.status_message
            else:
                msg.link_state = copy.deepcopy(self.link_map[key])
        return msg

    def update_tracker(self, delta_time, sim_time):
        """
        Update link_states of the links that this tracker is tracking

        Args:
            delta_time (float): delta time
            sim_time (Clock): simulation time
        """
        if self.link_names:
            with self.lock:
                res = self._get_link_states(self.link_names, self.reference_frames)
                if res.success:
                    for link_state, status in zip(res.link_states, res.status):
                        if status:
                            link_name = link_state.link_name
                            reference_frame = link_state.reference_frame
                            key = (link_name, reference_frame)
                            self.link_map[key] = link_state
