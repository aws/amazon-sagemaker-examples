'''This module houses the constants for the camera package'''
from enum import Enum, unique

# Define Gazebo World default direction unit vectors
class GazeboWorld(object):
    forward = (1.0, 0, 0)
    back = (-1.0, 0, 0)
    right = (0, -1.0, 0)
    left = (0, 1.0, 0)
    up = (0, 0, 1.0)
    down = (0, 0, -1.0)

@unique
class CameraSettings(Enum):
    '''This enum is used to index into the camera settings list'''
    HORZ_FOV = 1
    PADDING_PCT = 2
    IMG_WIDTH = 3
    IMG_HEIGHT = 4

    @classmethod
    def get_empty_dict(cls):
        '''Returns dictionary with the enum as key values and None's as the values, clients
           are responsible for populating the dict accordingly
        '''
        empty_dict = dict()
        for val in cls._value2member_map_.values():
            empty_dict[val] = None
        return empty_dict
