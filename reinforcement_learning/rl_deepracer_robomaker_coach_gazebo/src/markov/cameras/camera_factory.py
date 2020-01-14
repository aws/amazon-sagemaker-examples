from markov.cameras.handlers import FollowCarCamera, FollowTrackBehindCamera, FollowTrackOffsetCamera, TopCamera
from markov.deepracer_exceptions import GenericRolloutException
from enum import Enum


class CameraType(Enum):
    """
    Camera Type enum
    """
    FOLLOW_CAR_CAMERA = FollowCarCamera.name
    FOLLOW_TRACK_BEHIND_CAMERA = FollowTrackBehindCamera.name
    FOLLOW_TRACK_OFFSET_CAMERA = FollowTrackOffsetCamera.name
    TOP_CAMERA = TopCamera.name


"""
Camera Type to Camera Class map
"""
CAMERA_TYPE_TO_CAMERA_CLASS_MAP = {
    CameraType.FOLLOW_CAR_CAMERA: FollowCarCamera,
    CameraType.FOLLOW_TRACK_BEHIND_CAMERA: FollowTrackBehindCamera,
    CameraType.FOLLOW_TRACK_OFFSET_CAMERA: FollowTrackOffsetCamera,
    CameraType.TOP_CAMERA: TopCamera
}


class CameraFactory(object):
    """
    This class implements a camera factory and is used to create camera.
    """
    @staticmethod
    def get_class(camera_type):
        """
        Factory method for getting camera class
            camera_type - String containing the desired camera type
        """
        try:
            if isinstance(camera_type, str):
                camera_type = CameraType(camera_type)
            return CAMERA_TYPE_TO_CAMERA_CLASS_MAP[camera_type]
        except:
            raise GenericRolloutException("Unknown camera")

    @staticmethod
    def get_instance(camera_type):
        """
        Factory method for creating camera
            camera_type - String containing the desired camera type
        """
        camera_class = CameraFactory.get_class(camera_type)
        return camera_class.get_instance()
