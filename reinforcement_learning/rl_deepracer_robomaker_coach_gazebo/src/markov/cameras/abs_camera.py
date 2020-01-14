import abc
import os
import rospkg
import threading

from markov.deepracer_exceptions import GenericRolloutException
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.track_geom.constants import SPAWN_SDF_MODEL
from gazebo_msgs.srv import SpawnModel

# Python 2 and 3 compatible Abstract class
ABC = abc.ABCMeta('ABC', (object,), {})


class BaseCamera(ABC):
    """
    Abstract Camera method
    """
    def __init__(self, name):
        if not name or not isinstance(name, str):
            raise GenericRolloutException("Camera name cannot be None or empty string")
        self._name = name
        self._topic_name = self._name
        self.lock = threading.Lock()
        self.is_reset_called = False
        self.spawn_sdf_model = ServiceProxyWrapper(SPAWN_SDF_MODEL, SpawnModel)
        self.rospack = rospkg.RosPack()

    @property
    def name(self):
        """Return name of Camera

        Returns:
            (str): the name of camera
        """
        return self._name

    @property
    def topic_name(self):
        """Return name of gazebo topic

        Returns:
            (str): the name of camera
        """
        return self._topic_name

    @topic_name.setter
    def topic_name(self, val):
        """
        Set gazebo topic name

        Args:
            val: topic name
        """
        self._topic_name = val

    @classmethod
    def has_instance(cls):
        if not hasattr(cls, '_instance_'):
            raise GenericRolloutException("Camera class requires _instance_ static member variable")
        return cls._instance_ is not None

    def reset_pose(self, model_state):
        """
        Reset the camera pose

        Args:
            model_state (object): State object
        """
        with self.lock:
            self._reset(model_state)
            self.is_reset_called = True

    def spawn_model(self, model_state):
        deepracer_path = self.rospack.get_path("deepracer_simulation_environment")
        camera_sdf_path = os.path.join(deepracer_path, "models", "camera", "model.sdf")
        with open(camera_sdf_path, "r") as fp:
            camera_sdf = fp.read()
        camera_pose = self._get_initial_camera_pose(model_state)
        self.spawn_sdf_model(self.topic_name, camera_sdf, '/{}'.format(self.topic_name),
                             camera_pose, '')

    def update_pose(self, model_state, delta_time):
        """
        Update the camera pose

        Args:
            model_state (object): State object
            delta_time (float): time delta from last update
        """
        with self.lock:
            if self.is_reset_called:
                self._update(model_state, delta_time)

    @abc.abstractmethod
    def _reset(self, model_state):
        """
        Reset the camera pose
        """
        raise NotImplementedError('Camera must be able to reset')

    @abc.abstractmethod
    def _update(self, model_state, delta_time):
        """Update the camera pose
        """
        raise NotImplementedError('Camera must be able to update')

    @abc.abstractmethod
    def _get_initial_camera_pose(self, model_state):
        """compuate camera pose
        """
        raise NotImplementedError('Camera must be able to compuate pose')
        