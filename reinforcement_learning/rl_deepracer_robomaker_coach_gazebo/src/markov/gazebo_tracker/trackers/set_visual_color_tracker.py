import threading
from collections import OrderedDict

import markov.gazebo_tracker.constants as consts
import rospy
from deepracer_msgs.srv import SetVisualColorResponse, SetVisualColors, SetVisualColorsRequest
from markov.domain_randomizations.constants import GazeboServiceName
from markov.gazebo_tracker.abs_tracker import AbstractTracker
from markov.log_handler.deepracer_exceptions import GenericRolloutException
from markov.rospy_wrappers import ServiceProxyWrapper


class SetVisualColorTracker(AbstractTracker):
    """
    SetVisualColorTracker Tracker class
    """

    _instance_ = None

    @staticmethod
    def get_instance():
        """Method for getting a reference to the SetVisualColor Tracker object"""
        if SetVisualColorTracker._instance_ is None:
            SetVisualColorTracker()
        return SetVisualColorTracker._instance_

    def __init__(self):
        if SetVisualColorTracker._instance_ is not None:
            raise GenericRolloutException("Attempting to construct multiple SetVisualColor Tracker")

        self.lock = threading.RLock()
        self.visual_name_map = OrderedDict()
        self.link_name_map = OrderedDict()
        self.ambient_map = OrderedDict()
        self.diffuse_map = OrderedDict()
        self.specular_map = OrderedDict()
        self.emissive_map = OrderedDict()

        rospy.wait_for_service(GazeboServiceName.SET_VISUAL_COLORS.value)
        self.set_visual_colors = ServiceProxyWrapper(
            GazeboServiceName.SET_VISUAL_COLORS.value, SetVisualColors
        )

        SetVisualColorTracker._instance_ = self
        super(SetVisualColorTracker, self).__init__(priority=consts.TrackerPriority.LOW)

    def set_visual_color(
        self, visual_name, link_name, ambient, diffuse, specular, emissive, blocking=False
    ):
        """
        Set Material that will be updated in next update call
        Args:
            visual_name (str): name of visual
            link_name (str):  name of the link holding visual
            ambient (ColorRBGA): ambient color
            diffuse (ColorRBGA): diffuse color
            specular (ColorRBGA): specular color
            emissive (ColorRBGA): emissive color
            blocking (bool): flag to block or not
        Returns:
            msg (SetVisualColorResponse)
        """
        msg = SetVisualColorResponse()
        msg.success = True
        key = (visual_name, link_name)
        with self.lock:
            if blocking:
                if key in self.visual_name_map:
                    del self.visual_name_map[key]
                    del self.link_name_map[key]
                    del self.ambient_map[key]
                    del self.diffuse_map[key]
                    del self.specular_map[key]
                    del self.emissive_map[key]

                req = SetVisualColorsRequest()
                req.visual_names = [visual_name]
                req.link_names = [link_name]
                req.ambients = [ambient]
                req.diffuses = [diffuse]
                req.speculars = [specular]
                req.emissives = [emissive]
                res = self.set_visual_colors(req)
                msg.success = res.success and res.status[0]
                msg.status_message = res.messages[0] if res.success else res.status_message
            else:
                self.visual_name_map[key] = visual_name
                self.link_name_map[key] = link_name
                self.ambient_map[key] = ambient
                self.diffuse_map[key] = diffuse
                self.specular_map[key] = specular
                self.emissive_map[key] = emissive
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
                req = SetVisualColorsRequest()

                req.visual_names = self.visual_name_map.values()
                req.link_names = self.link_name_map.values()
                req.ambients = self.ambient_map.values()
                req.diffuses = self.diffuse_map.values()
                req.speculars = self.specular_map.values()
                req.emissives = self.emissive_map.values()
                self.set_visual_colors(req)

            self.visual_name_map = OrderedDict()
            self.link_name_map = OrderedDict()
            self.ambient_map = OrderedDict()
            self.diffuse_map = OrderedDict()
            self.specular_map = OrderedDict()
            self.emissive_map = OrderedDict()
