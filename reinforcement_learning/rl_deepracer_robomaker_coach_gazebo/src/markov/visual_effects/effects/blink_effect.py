import rospy
from deepracer_msgs.srv import (
    GetVisualNames,
    GetVisualNamesRequest,
    GetVisuals,
    GetVisualsRequest,
    GetVisualsResponse,
)
from gazebo_msgs.srv import GetModelProperties, GetModelPropertiesRequest
from markov.cameras.utils import lerp
from markov.domain_randomizations.constants import GazeboServiceName
from markov.gazebo_tracker.trackers.set_visual_transparency_tracker import (
    SetVisualTransparencyTracker,
)
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.track_geom.track_data import TrackData
from markov.visual_effects.abs_effect import AbstractEffect


class BlinkEffect(AbstractEffect):
    def __init__(self, model_name, min_alpha=0.3, max_alpha=1.0, interval=1.0, duration=2.0):
        """
        Constructor

        Args:
            model_name (str): name of the model
            min_alpha (float): minimum alpha for blink
            max_alpha (float): maximum alpha for blink
            interval (float): interval in second for one cycle of blink
            duration (float): effect duration in second
        """
        super(BlinkEffect, self).__init__()
        self.model_name = model_name
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.interval = interval
        self.duration = duration

        self.source_alpha = self.min_alpha
        self.target_alpha = self.max_alpha
        self.current_interval = 0.0
        self.current_duration = 0.0

    def reset(self):
        """
        Reset internal variables
        """
        self.source_alpha = self.min_alpha
        self.target_alpha = self.max_alpha
        self.current_interval = 0.0
        self.current_duration = 0.0

    def _lazy_init(self):
        """
        Lazy-initialize effect
        """
        # ROS Services Setup
        rospy.wait_for_service(GazeboServiceName.GET_MODEL_PROPERTIES.value)
        rospy.wait_for_service(GazeboServiceName.GET_VISUAL_NAMES.value)
        rospy.wait_for_service(GazeboServiceName.GET_VISUALS.value)

        get_model_prop = ServiceProxyWrapper(
            GazeboServiceName.GET_MODEL_PROPERTIES.value, GetModelProperties
        )
        get_visual_names = ServiceProxyWrapper(
            GazeboServiceName.GET_VISUAL_NAMES.value, GetVisualNames
        )
        get_visuals = ServiceProxyWrapper(GazeboServiceName.GET_VISUALS.value, GetVisuals)

        # Get all model's link names
        body_names = get_model_prop(
            GetModelPropertiesRequest(model_name=self.model_name)
        ).body_names
        link_names = ["%s::%s" % (self.model_name, b) for b in body_names]

        res = get_visual_names(GetVisualNamesRequest(link_names=link_names))
        get_visuals_req = GetVisualsRequest(
            link_names=res.link_names, visual_names=res.visual_names
        )
        self.orig_visuals = get_visuals(get_visuals_req)

    def on_attach(self):
        """
        During attach, add model to non-collidable objects.
        """
        self.reset()
        # Add to noncollidable_objects to avoid collision with other objects during blink
        TrackData.get_instance().add_noncollidable_object(self.model_name)

    def on_detach(self):
        """
        After detach, remove model from non-collidable objects and reset transparencies to original.
        """
        for visual_name, link_name, transparency in zip(
            self.orig_visuals.visual_names,
            self.orig_visuals.link_names,
            [0.0] * len(self.orig_visuals.visual_names),
        ):
            SetVisualTransparencyTracker.get_instance().set_visual_transparency(
                visual_name, link_name, transparency
            )

        # Remove noncollidable_objects to allow collision with other objects after blink
        TrackData.get_instance().remove_noncollidable_object(self.model_name)

    def _update(self, delta_time):
        """
        Update blink effect

        Args:
            delta_time (float): the change of time in second from last call
        """
        if self.current_duration < self.duration:
            cur_alpha = lerp(
                self.source_alpha, self.target_alpha, self.current_interval / self.interval
            )
            transparencies = [1.0 - cur_alpha for _ in self.orig_visuals.transparencies]
            for visual_name, link_name, transparency in zip(
                self.orig_visuals.visual_names, self.orig_visuals.link_names, transparencies
            ):
                SetVisualTransparencyTracker.get_instance().set_visual_transparency(
                    visual_name, link_name, transparency
                )

            self.current_interval += delta_time
            if self.current_interval >= self.interval:
                temp = self.source_alpha
                self.source_alpha = self.target_alpha
                self.target_alpha = temp
                self.current_interval = 0.0
            self.current_duration += delta_time
        else:
            self.detach()
