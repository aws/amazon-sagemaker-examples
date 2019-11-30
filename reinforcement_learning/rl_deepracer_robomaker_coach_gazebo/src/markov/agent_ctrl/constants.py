'''This module houses the constants for the agent ctlr package'''
from enum import Enum

# Default max number of steps to allow per episode
MAX_STEPS = 10000

# Local offset of the front of the car
RELATIVE_POSITION_OF_FRONT_OF_CAR = [0.14, 0, 0]

# Normalized track distance to move with each reset
ROUND_ROBIN_ADVANCE_DIST = 0.05

# Reward to give the car when it "crashes"
CRASHED = 1e-8
# The number of steps to wait before checking if the car is stuck
# This number should corespond to the camera FPS, since it is pacing the
# step rate.
NUM_STEPS_TO_CHECK_STUCK = 15

# Radius of the wheels of the car in meters
WHEEL_RADIUS = 0.1

class ConfigParams(Enum):
    '''This enum defines the keys for the input keys for the rollout
       ctr config dict
    '''
    AGENT_NAME = 'agent_name'
    LINK_NAME_LIST = 'agent_link_name_list'
    STEERING_LIST = 'steering_list'
    VELOCITY_LIST = 'velocity_list'
    REWARD = 'reward'
    ACTION_SPACE_PATH = 'path_to_json'
    CHANGE_START = 'change_start'
    ALT_DIR = 'alternate_dir'
    VERSION = 'version'

class RewardParam(Enum):
    '''This enum contains the keys and default values for the parameters to be
       feed into the reward function.
    '''
    # boolean: all wheel on track
    WHEELS_ON_TRACK = ['all_wheels_on_track', True]
    X = ['x', 0.0]                                                      # float: race car x position
    Y = ['y', 0.0]                                                      # float: race car y position
    HEADING = ['heading', 0.0]                                          # float: race car heading angle
    CENTER_DIST = ['distance_from_center', 0.0]                         # float: race car distance from centerline
    PROG = ['progress', 0.0]                                            # float: race car track progress [0,1]
    STEPS = ['steps', 0]                                                # int: number of steps race car have taken
    SPEED = ['speed', 0.0]                                              # float: race car speed
    STEER = ['steering_angle', 0.0]                                     # float: race car steering angle
    TRACK_WIDTH = ['track_width', 0.0]                                  # float: track width
    TRACK_LEN = ['track_length', 0.0]                                   # float: track length
    WAYPNTS = ['waypoints', 0]                                          # list of tuple: list of waypoints (x, y) tuple
    CLS_WAYPNY = ['closest_waypoints', [0, 0]]                          # list of int: list of int with size 2 containing closest prev and next waypoint indexes
    LEFT_CENT = ['is_left_of_center', False]                            # boolean: race car left of centerline
    REVERSE = ['is_reversed', False]                                    # boolean: race car direction. True (clockwise), False (counterclockwise)
    CLOSEST_OBJECTS = ['closest_objects', [0, 0]]                       # list of int: list of int with size 2 containing closest prev and next object indexes
    OBJECT_LOCATIONS = ['objects_location', []]                         # list of tuple: list of all object (x, y) locations
    OBJECTS_LEFT_OF_CENTER = ['objects_left_of_center', []]             # list of boolean: list of all object to the left of centerline or not
    OBJECT_IN_CAMERA = ['object_in_camera', False]                      # boolean: any object in camera
    OBJECT_SPEEDS = ['objects_speed', []]                               # list of float: list of objects speed
    OBJECT_HEADINGS = ['objects_heading', []]                           # list of float: list of objects heading
    OBJECT_CENTERLINE_PROJECTION_DISTANCES = ['objects_distance', []]   # list of float: list of object distance projected on the centerline
    CRASHED = ['crashed', False]                                        # boolean: crashed into an object or bot car
    OFFTRACK = ['offtrack', False]                                      # boolean: all four wheels went off-track

    @classmethod
    def make_default_param(cls):
        '''Returns a dictionary with the default values for the reward function'''
        return {key.value[0] : key.value[-1] for key in cls}

    @classmethod
    def validate_dict(cls, input_dict):
        '''Will raise an exception if input dict does not contain all the keys in the enum'''
        for key in cls:
            _ = input_dict[key.value[0]]
