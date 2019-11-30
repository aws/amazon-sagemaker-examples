'''This module houses the constants for the camera package'''


# Define Gazebo World default direction unit vectors
class GazeboWorld(object):
    forward = (1.0, 0, 0)
    back = (-1.0, 0, 0)
    right = (0, -1.0, 0)
    left = (0, 1.0, 0)
    up = (0, 0, 1.0)
    down = (0, 0, -1.0)
