import abc
import threading

from markov.deepracer_exceptions import GenericRolloutException
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.track_geom.constants import SPAWN_SDF_MODEL
from markov.cameras.camera_manager import CameraManager
from gazebo_msgs.srv import SpawnModel

# Python 2 and 3 compatible Abstract class
ABC = abc.ABCMeta('ABC', (object,), {})


class AbstractCamera(ABC):
    """
    Abstract Camera method
    """
    def __init__(self, name, namespace, topic_name):
        if not name or not isinstance(name, str):
            raise GenericRolloutException("Camera name cannot be None or empty string")
        self._name = name
        self._topic_name = topic_name or self._name
        self._namespace = namespace or 'default'
        self.lock = threading.Lock()
        self.is_reset_called = False
        self.spawn_sdf_model = ServiceProxyWrapper(SPAWN_SDF_MODEL, SpawnModel)
        CameraManager.get_instance().add(self, namespace)

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

    @property
    def namespace(self):
        """Return namespace of camera in camera manager

        Returns:
            (str): the namespace of camera in camera manager
        """
        return self._namespace

    def reset_pose(self, model_state):
        """
        Reset the camera pose

        Args:
            model_state (object): State object
        """
        with self.lock:
            self._reset(model_state)
            self.is_reset_called = True

    def spawn_model(self, model_state, camera_sdf_path):
        """
        Spawns a sdf model located in the given path

        Args:
            model_state (object): State object
            camera_sdf_path (string): full path to the location of sdf file
        """
        camera_sdf = self._get_sdf_string(camera_sdf_path)
        camera_pose = self._get_initial_camera_pose(model_state)
        self.spawn_sdf_model(self.topic_name, camera_sdf, self.topic_name, camera_pose, '')

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
    def _get_sdf_string(self, camera_sdf_path):
        """
        Reads the sdf file and converts it to a string in
        memory

        Args:
            camera_sdf_path (string): full path to the location of sdf file
        """
        raise NotImplementedError('Camera must read and convert model sdf file')

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
