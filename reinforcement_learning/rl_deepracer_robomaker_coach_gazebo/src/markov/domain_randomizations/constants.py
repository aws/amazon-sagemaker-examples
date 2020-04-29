from enum import Enum


class GazeboServiceName(Enum):
    """Gazebo Service Names"""
    GET_MODEL_PROPERTIES = '/gazebo/get_model_properties'
    GET_LIGHT_NAMES = '/gazebo/get_light_names'
    GET_VISUAL_NAMES = '/gazebo/get_visual_names'
    GET_VISUAL_COLOR = '/gazebo/get_visual_color'
    GET_VISUAL_COLORS = '/gazebo/get_visual_colors'
    SET_VISUAL_COLOR = '/gazebo/set_visual_color'
    SET_VISUAL_COLORS = '/gazebo/set_visual_colors'
    SET_LIGHT_PROPERTIES = '/gazebo/set_light_properties'


class ModelRandomizerType(Enum):
    """ Model Randomizer Type

    MODEL type will randomize the color of overall model.
    LINK type will randomize the color for each link.
    VISUAL type will randomize the color for each link's visual
    """
    MODEL = "model"
    LINK = "link"
    VISUAL = "visual"


class RangeType(Enum):
    """Range Type"""
    COLOR = 'color'
    ATTENUATION = 'attenuation'


RANGE_MIN = 'min'
RANGE_MAX = 'max'


class Color(Enum):
    """Color attributes"""
    R = 'r'
    G = 'g'
    B = 'b'


class Attenuation(Enum):
    """Light attenuation attributes"""
    CONSTANT = 'constant'
    LINEAR = 'linear'
    QUADRATIC = 'quadratic'


