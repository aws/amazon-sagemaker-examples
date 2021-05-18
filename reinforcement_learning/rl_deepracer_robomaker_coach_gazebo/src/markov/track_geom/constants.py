"""This module contains all the constants for the track geom package"""
from enum import Enum, unique

GET_LINK_STATE = "/gazebo/get_link_state"
GET_MODEL_STATE = "/gazebo/get_model_state"
SET_MODEL_STATE = "/gazebo/set_model_state"
GET_LINK_STATES = "/gazebo/get_link_states"
GET_MODEL_STATES = "/gazebo/get_model_states"
SET_MODEL_STATES = "/gazebo/set_model_states"
SPAWN_SDF_MODEL = "/gazebo/spawn_sdf_model"
SPAWN_URDF_MODEL = "/gazebo/spawn_urdf_model"

SPLINE_DEGREE = 3

DIST_THRESHOLD = 0.02

START_POS_OFFSET = 0.5

# The HIDE_POS_* are set such that the cars that are not in use
# are hided outside of the world area of gazebo.
# This has been shown to boost the rtf value during simulation.
# The value is empirically decided to be a large enough one
# for the current track worlds.
HIDE_POS_OFFSET = 60.0

HIDE_POS_DELTA = 0.5


class TrackLane(Enum):
    INNER_LANE = "inner_lane"
    OUTER_LANE = "outer_lane"
    CENTER_LINE = "center_line"


class TrackNearPnts(Enum):
    """Keys for nearest points dictionary"""

    NEAR_PNT_CENT = "near_pnt_cent"
    NEAR_PNT_IN = "near_pnt_in"
    NEAR_PNT_OUT = "near_pnt_out"


class TrackNearDist(Enum):
    """Keys for nearest distance dictionary"""

    NEAR_DIST_CENT = "near_dist_cent"
    NEAR_DIST_IN = "near_dist_in"
    NEAR_DIST_OUT = "near_dist_out"


class AgentPos(Enum):
    """Keys for agent position dictionary"""

    ORIENTATION = "model_orientation"
    POINT = "model_point"
    LINK_POINTS = "link_points"


class ObstacleDimensions(Enum):
    """The dimensions of different obstacle"""

    BOX_OBSTACLE_DIMENSION = (0.4, 0.5)  # Length(x), width(y) + 0.1 buffer for each
    # TODO: Fix the bot car dimension so that all collisions are detected
    BOT_CAR_DIMENSION = (0.296, 0.422)  # Length(x), width(y) + 0.1 buffer for each


class ParkLocation(Enum):
    """Park location type"""

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
