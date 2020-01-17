'''This module contains all the constants for the track geom package'''
from enum import Enum, unique

GET_LINK_STATE = '/gazebo/get_link_state'
GET_MODEL_STATE = '/gazebo/get_model_state'
SET_MODEL_STATE = '/gazebo/set_model_state'
SPAWN_SDF_MODEL = '/gazebo/spawn_sdf_model'
SPAWN_URDF_MODEL = '/gazebo/spawn_urdf_model'

class TrackNearPnts(Enum):
    '''Keys for nearest points dictionary'''
    NEAR_PNT_CENT = 'near_pnt_cent'
    NEAR_PNT_IN = 'near_pnt_in'
    NEAR_PNT_OUT = 'near_pnt_out'

class TrackNearDist(Enum):
    '''Keys for nearest distance dictionary'''
    NEAR_DIST_CENT = 'near_dist_cent'
    NEAR_DIST_IN = 'near_dist_in'
    NEAR_DIST_OUT = 'near_dist_out'

class AgentPos(Enum):
    '''Keys for agent position dictionary'''
    ORIENTATION = 'model_orientation'
    POINT = 'model_point'
    LINK_POINTS = 'link_points'

class ObstacleDimensions(Enum):
    ''' The dimensions of different obstacle '''
    BOX_OBSTACLE_DIMENSION = (0.4, 0.5) # Length(x), width(y) + 0.1 buffer for each
    BOT_CAR_DIMENSION = (0.296, 0.422) # Length(x), width(y) + 0.1 buffer for each