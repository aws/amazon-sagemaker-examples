import logging

import numpy as np
import rospy
from deepracer_msgs.srv import GetVisualNames, GetVisualNamesRequest
from gazebo_msgs.srv import GetModelProperties, GetModelPropertiesRequest
from markov.domain_randomizations.abs_randomizer import AbstractRandomizer
from markov.domain_randomizations.constants import (
    RANGE_MAX,
    RANGE_MIN,
    Color,
    GazeboServiceName,
    ModelRandomizerType,
)
from markov.gazebo_tracker.trackers.set_visual_color_tracker import SetVisualColorTracker
from markov.log_handler.logger import Logger
from markov.rospy_wrappers import ServiceProxyWrapper
from std_msgs.msg import ColorRGBA

logger = Logger(__name__, logging.INFO).get_logger()


class ModelVisualRandomizer(AbstractRandomizer):
    """Model Visual Randomizer class"""

    def __init__(
        self,
        model_name,
        model_randomizer_type,
        num_selection=-1,
        link_name_filter=None,
        visual_name_filter=None,
        color_range=None,
    ):
        """
        Constructor
        - Bit of explanation regarding model_randomizer_type:
            - There are 3 possible types (MODEL, LINK, VISUAL) due to the level of randomization.
              The reason is that a model's visual is represented in three level and their relationship as below:
              - 1 Model to N Links and 1 Link to M Visuals.
            - The one case that LINK may contain multiple VISUALs is that gazebo merges the links
              that is connected by fixed joint for the sake of physics performance. Even the links are merged
              together, it still needs to keep the visuals separately to display correctly with its own materials.
              Thus, single merged link can contain multiple visuals from the links before the merge.

        Args:
            model_name (str): name of the model
            model_randomizer_type (ModelRandomizerType): Model Randomizer Type
            num_selection (int): Number of visuals or link to select on each randomize. (-1 means all)
                                 (Only used for ModelRandomizerType.LINK or ModelRandomizerType.VISUAL)
            link_name_filter (set or list): If link_name_filter are provided,
                                            randomization will only apply to given links.
            visual_name_filter (set or list): If visual_name_filter are provided,
                                              randomization will only apply to given visuals.
            color_range (dict): min-max of each color component (r, g, b).
                Valid format: {'r': {'min':0.0, 'max':1.0},
                               'g': {'min':0.0, 'max':1.0},
                               'b': {'min':0.0, 'max':1.0}}
        """
        super(ModelVisualRandomizer, self).__init__()
        self.model_name = model_name
        self.model_randomizer_type = model_randomizer_type
        self.num_selection = num_selection

        self.color_range = {
            Color.R.value: {RANGE_MIN: 0.0, RANGE_MAX: 1.0},
            Color.G.value: {RANGE_MIN: 0.0, RANGE_MAX: 1.0},
            Color.B.value: {RANGE_MIN: 0.0, RANGE_MAX: 1.0},
        }
        if color_range:
            self.color_range.update(color_range)

        # ROS Services Setup
        rospy.wait_for_service(GazeboServiceName.GET_MODEL_PROPERTIES.value)
        rospy.wait_for_service(GazeboServiceName.GET_VISUAL_NAMES.value)

        get_model_prop = ServiceProxyWrapper(
            GazeboServiceName.GET_MODEL_PROPERTIES.value, GetModelProperties
        )
        get_visual_names = ServiceProxyWrapper(
            GazeboServiceName.GET_VISUAL_NAMES.value, GetVisualNames
        )

        # Get all model's link names
        body_names = get_model_prop(
            GetModelPropertiesRequest(model_name=self.model_name)
        ).body_names
        link_names = ["%s::%s" % (model_name, b) for b in body_names]

        # Convert filters to sets
        link_name_filter = set(link_name_filter) if link_name_filter is not None else None
        visual_name_filter = set(visual_name_filter) if visual_name_filter is not None else None

        if link_name_filter is not None:
            # If link_name_filter is not None then grab the link_name that is in link_name_filter only.
            link_names = [link_name for link_name in link_names if link_name in link_name_filter]

        self.link_visuals_map = {}
        res = get_visual_names(GetVisualNamesRequest(link_names=link_names))
        for idx, visual_name in enumerate(res.visual_names):
            if visual_name_filter is not None and visual_name not in visual_name_filter:
                continue
            link_name = res.link_names[idx]
            if link_name not in self.link_visuals_map:
                self.link_visuals_map[link_name] = []
            self.link_visuals_map[link_name].append(visual_name)
        # logger.info('link_visuals_map: {}'.format({"model_name:": self.model_name, "links": self.link_visuals_map}))

    def _get_random_color(self):
        return ColorRGBA(
            *[
                np.random.uniform(
                    self.color_range[Color.R.value][RANGE_MIN],
                    self.color_range[Color.R.value][RANGE_MAX],
                ),
                np.random.uniform(
                    self.color_range[Color.G.value][RANGE_MIN],
                    self.color_range[Color.G.value][RANGE_MAX],
                ),
                np.random.uniform(
                    self.color_range[Color.B.value][RANGE_MIN],
                    self.color_range[Color.B.value][RANGE_MAX],
                ),
                1.0,
            ]
        )

    def _randomize(self):
        link_names = self.link_visuals_map.keys()
        # Unroll all visual names
        visual_names = [
            visual_name
            for visual_names in self.link_visuals_map.values()
            for visual_name in visual_names
        ]

        if self.model_randomizer_type == ModelRandomizerType.LINK and self.num_selection > 0:
            # Select links to randomize if model_randomizer_type is ModelRandomizerType.LINK
            link_names = np.random.choice(
                self.link_visuals_map.keys(), size=self.num_selection, replace=False
            )
        elif self.model_randomizer_type == ModelRandomizerType.VISUAL and self.num_selection > 0:
            # Select visuals to randomize if model_randomizer_type is ModelRandomizerType.VISUAL
            visual_names = np.random.choice(visual_names, size=self.num_selection, replace=False)
        # Convert to set
        visual_names = set(visual_names)
        # Model-level random color
        color = self._get_random_color()
        ambient = color
        diffuse = color
        specular = ColorRGBA(0.0, 0.0, 0.0, 1.0)
        emissive = ColorRGBA(0.0, 0.0, 0.0, 1.0)

        for link_name in link_names:
            for idx, visual_name in enumerate(self.link_visuals_map[link_name]):
                if visual_name not in visual_names:
                    continue
                SetVisualColorTracker.get_instance().set_visual_color(
                    visual_name=visual_name,
                    link_name=link_name,
                    ambient=ambient,
                    diffuse=diffuse,
                    specular=specular,
                    emissive=emissive,
                )

                if self.model_randomizer_type == ModelRandomizerType.VISUAL:
                    # Visual-level random color
                    color = self._get_random_color()
                    ambient = color
                    diffuse = color

            if self.model_randomizer_type == ModelRandomizerType.LINK:
                # Link-level random color
                color = self._get_random_color()
                ambient = color
                diffuse = color
