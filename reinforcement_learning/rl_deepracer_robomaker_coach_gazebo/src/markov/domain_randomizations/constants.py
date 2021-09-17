from enum import Enum


class GeometryType(Enum):
    """Geometry Type"""

    BOX = 1
    CYLINDER = 2
    SPHERE = 3
    PLANE = 4
    IMAGE = 5
    HEIGHTMAP = 6
    MESH = 7
    TRIANGLE_FAN = 8
    LINE_STRIP = 9
    POLYLINE = 10
    EMPTY = 11


class GazeboServiceName(Enum):
    """Gazebo Service Names"""

    GET_MODEL_PROPERTIES = "/gazebo/get_model_properties"
    GET_LIGHT_NAMES = "/gazebo/get_light_names"
    GET_VISUAL_NAMES = "/gazebo/get_visual_names"
    GET_VISUAL = "/gazebo/get_visual"
    GET_VISUALS = "/gazebo/get_visuals"
    SET_LIGHT_PROPERTIES = "/gazebo/set_light_properties"
    SET_VISUAL_COLOR = "/gazebo/set_visual_color"
    SET_VISUAL_COLORS = "/gazebo/set_visual_colors"
    SET_VISUAL_TRANSPARENCY = "/gazebo/set_visual_transparency"
    SET_VISUAL_TRANSPARENCIES = "/gazebo/set_visual_transparencies"
    SET_VISUAL_VISIBLE = "/gazebo/set_visual_visible"
    SET_VISUAL_VISIBLES = "/gazebo/set_visual_visibles"
    SET_VISUAL_POSE = "/gazebo/set_visual_pose"
    SET_VISUAL_POSES = "/gazebo/set_visual_poses"
    SET_VISUAL_MESH = "/gazebo/set_visual_mesh"
    SET_VISUAL_MESHES = "/gazebo/set_visual_meshes"
    PAUSE_PHYSICS = "/gazebo/pause_physics_dr"
    UNPAUSE_PHYSICS = "/gazebo/unpause_physics_dr"


class ModelRandomizerType(Enum):
    """Model Randomizer Type

    MODEL type will randomize the color of overall model.
    LINK type will randomize the color for each link.
    VISUAL type will randomize the color for each link's visual
    """

    MODEL = "model"
    LINK = "link"
    VISUAL = "visual"


class RangeType(Enum):
    """Range Type"""

    COLOR = "color"
    ATTENUATION = "attenuation"


RANGE_MIN = "min"
RANGE_MAX = "max"


class Color(Enum):
    """Color attributes"""

    R = "r"
    G = "g"
    B = "b"


class Attenuation(Enum):
    """Light attenuation attributes"""

    CONSTANT = "constant"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
