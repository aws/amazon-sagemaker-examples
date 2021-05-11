import threading
from collections import OrderedDict

import markov.gazebo_tracker.constants as consts
import rospy
from deepracer_msgs.srv import (
    SetVisualTransparencies,
    SetVisualTransparenciesRequest,
    SetVisualTransparencyResponse,
)
from markov.domain_randomizations.constants import GazeboServiceName
from markov.gazebo_tracker.abs_tracker import AbstractTracker
from markov.log_handler.deepracer_exceptions import GenericRolloutException
from markov.rospy_wrappers import ServiceProxyWrapper


class SetVisualTransparencyTracker(AbstractTracker):
    """
    SetVisualTransparency Tracker class
    """

    _instance_ = None

    @staticmethod
    def get_instance():
        """Method for getting a reference to the SetVisualTransparency Tracker object"""
        if SetVisualTransparencyTracker._instance_ is None:
            SetVisualTransparencyTracker()
        return SetVisualTransparencyTracker._instance_

    def __init__(self):
        if SetVisualTransparencyTracker._instance_ is not None:
            raise GenericRolloutException(
                "Attempting to construct multiple SetVisualTransparency Tracker"
            )

        self.lock = threading.RLock()
        self.visual_name_map = OrderedDict()
        self.link_name_map = OrderedDict()
        self.transparency_map = OrderedDict()

        rospy.wait_for_service(GazeboServiceName.SET_VISUAL_TRANSPARENCIES.value)
        self.set_visual_transparencies = ServiceProxyWrapper(
            GazeboServiceName.SET_VISUAL_TRANSPARENCIES.value, SetVisualTransparencies
        )

        SetVisualTransparencyTracker._instance_ = self
        super(SetVisualTransparencyTracker, self).__init__(priority=consts.TrackerPriority.LOW)

    def set_visual_transparency(self, visual_name, link_name, transparency, blocking=False):
        """
        Set transparency that will be updated in next update call
        Args:
            visual_name (str): name of visual
            link_name (str):  name of the link holding visual
            transparency (float): visual's transparency between 0.0 (opaque) and 1.0 (full transparent)
            blocking (bool): flag to block or not
        Returns:
            msg (SetVisualTransparencyResponse)
        """
        msg = SetVisualTransparencyResponse()
        key = (visual_name, link_name)
        with self.lock:
            if blocking:
                if key in self.visual_name_map:
                    del self.visual_name_map[key]
                    del self.link_name_map[key]
                    del self.transparency_map[key]
                req = SetVisualTransparenciesRequest()
                req.visual_names = [visual_name]
                req.link_names = [link_name]
                req.transparencies = [transparency]
                res = self.set_visual_transparencies(req)
                msg.success = res.success and res.status[0]
                msg.status_message = res.messages[0] if res.success else res.status_message
            else:
                self.visual_name_map[key] = visual_name
                self.link_name_map[key] = link_name
                self.transparency_map[key] = transparency
        return msg

    def update_tracker(self, delta_time, sim_time):
        """
        Update all color materials tracking to gazebo

        Args:
            delta_time (float): delta time
            sim_time (Clock): simulation time
        """
        with self.lock:
            if self.visual_name_map.values():
                req = SetVisualTransparenciesRequest()

                req.visual_names = self.visual_name_map.values()
                req.link_names = self.link_name_map.values()
                req.transparencies = self.transparency_map.values()
                self.set_visual_transparencies(req)

            self.visual_name_map = OrderedDict()
            self.link_name_map = OrderedDict()
            self.transparency_map = OrderedDict()
