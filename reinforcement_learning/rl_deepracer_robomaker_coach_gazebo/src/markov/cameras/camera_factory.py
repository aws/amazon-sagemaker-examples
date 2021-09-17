from enum import Enum

from markov.cameras.handlers import FollowCarCamera, TopCamera
from markov.log_handler.deepracer_exceptions import GenericRolloutException


class CameraType(Enum):
    """
    Camera Type enum
    """

    FOLLOW_CAR_CAMERA = FollowCarCamera.name
    TOP_CAMERA = TopCamera.name


class CameraFactory(object):
    """
    This class implements a camera factory and is used to create camera.
    """

    @staticmethod
    def create_instance(camera_type, namespace=None, model_name=None):
        """
        Factory method for creating camera instance
            camera_type - Enum type or String containing the desired camera type
        """
        try:
            if isinstance(camera_type, str):
                camera_type = CameraType(camera_type)
        except:
            raise GenericRolloutException("Unknown camera type: {}".format(camera_type))

        if camera_type == CameraType.FOLLOW_CAR_CAMERA:
            return FollowCarCamera(namespace=namespace, model_name=model_name)
        elif camera_type == CameraType.TOP_CAMERA:
            return TopCamera(namespace=namespace, model_name=model_name)
        else:
            raise GenericRolloutException("Unknown defined camera type: {}".format(camera_type))
