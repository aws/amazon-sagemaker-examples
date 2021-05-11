import logging
from enum import Enum

import numpy as np
import rospy
from gazebo_msgs.srv import SetLightProperties, SetLightPropertiesRequest
from markov.domain_randomizations.abs_randomizer import AbstractRandomizer
from markov.domain_randomizations.constants import (
    RANGE_MAX,
    RANGE_MIN,
    Attenuation,
    Color,
    GazeboServiceName,
    RangeType,
)
from markov.log_handler.logger import Logger
from markov.rospy_wrappers import ServiceProxyWrapper
from std_msgs.msg import ColorRGBA

logger = Logger(__name__, logging.INFO).get_logger()


class LightRandomizer(AbstractRandomizer):
    """Light Randomizer class"""

    def __init__(self, light_name, color_range=None, attenuation_range=None):
        """
        Constructor

        Args:
            light_name (str): name of the light
            color_range (dict): min-max of each color component (r, g, b).
                Valid format: {'r': {'min': 0.0, 'max': 1.0},
                               'g': {'min': 0.0, 'max': 1.0},
                               'b': {'min': 0.0, 'max': 1.0}}
            attenuation_range (dict): min-max of each attenuation component (constant, linear, quadratic).
                Valid format: {'constant': {'min': 0.0, 'max':1.0},
                               'linear': {'min': 0.0, 'max':1.0},
                               'quadratic': {'min': 0.0, 'max':1.0}}
        """
        super(LightRandomizer, self).__init__()
        self.light_name = light_name

        self.range = {
            RangeType.COLOR: {
                Color.R.value: {RANGE_MIN: 0.0, RANGE_MAX: 1.0},
                Color.G.value: {RANGE_MIN: 0.0, RANGE_MAX: 1.0},
                Color.B.value: {RANGE_MIN: 0.0, RANGE_MAX: 1.0},
            },
            RangeType.ATTENUATION: {
                Attenuation.CONSTANT.value: {RANGE_MIN: 0.0, RANGE_MAX: 1.0},
                Attenuation.LINEAR.value: {RANGE_MIN: 0.0, RANGE_MAX: 1.0},
                Attenuation.QUADRATIC.value: {RANGE_MIN: 0.0, RANGE_MAX: 1.0},
            },
        }
        if color_range:
            self.range[RangeType.COLOR].update(color_range)
        if attenuation_range:
            self.range[RangeType.ATTENUATION].update(attenuation_range)

        # ROS Services
        rospy.wait_for_service(GazeboServiceName.SET_LIGHT_PROPERTIES.value)
        self.set_light_prop = ServiceProxyWrapper(
            GazeboServiceName.SET_LIGHT_PROPERTIES.value, SetLightProperties
        )

    def _randomize(self):
        req = SetLightPropertiesRequest()
        req.light_name = self.light_name

        color_range = self.range[RangeType.COLOR]
        req.diffuse = ColorRGBA(
            *[
                np.random.uniform(
                    color_range[Color.R.value][RANGE_MIN], color_range[Color.R.value][RANGE_MAX]
                ),
                np.random.uniform(
                    color_range[Color.G.value][RANGE_MIN], color_range[Color.G.value][RANGE_MAX]
                ),
                np.random.uniform(
                    color_range[Color.B.value][RANGE_MIN], color_range[Color.B.value][RANGE_MAX]
                ),
                1.0,
            ]
        )

        attenuation_range = self.range[RangeType.ATTENUATION]
        req.attenuation_constant = np.random.uniform(
            attenuation_range[Attenuation.CONSTANT.value][RANGE_MIN],
            attenuation_range[Attenuation.CONSTANT.value][RANGE_MAX],
        )
        req.attenuation_linear = np.random.uniform(
            attenuation_range[Attenuation.LINEAR.value][RANGE_MIN],
            attenuation_range[Attenuation.LINEAR.value][RANGE_MAX],
        )
        req.attenuation_quadratic = np.random.uniform(
            attenuation_range[Attenuation.QUADRATIC.value][RANGE_MIN],
            attenuation_range[Attenuation.QUADRATIC.value][RANGE_MAX],
        )
        res = self.set_light_prop(req)
