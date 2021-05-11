import abc
import threading

from gazebo_msgs.srv import SpawnModel
from markov.cameras.camera_manager import CameraManager
from markov.log_handler.deepracer_exceptions import GenericRolloutException
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.track_geom.constants import SPAWN_SDF_MODEL

# Python 2 and 3 compatible Abstract class
ABC = abc.ABCMeta("ABC", (object,), {})


class AbstractCamera(ABC):
    """
    Abstract Camera method
    """

    def __init__(self, name, namespace, model_name):
        if not name or not isinstance(name, str):
            raise GenericRolloutException("Camera name cannot be None or empty string")
        self._name = name
        self._model_name = model_name or self._name
        self._namespace = namespace or "default"
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
    def model_name(self):
        """Return name of gazebo topic

        Returns:
            (str): the name of camera
        """
        return self._model_name

    @property
    def namespace(self):
        """Return namespace of camera in camera manager

        Returns:
            (str): the namespace of camera in camera manager
        """
        return self._namespace

    def reset_pose(self, car_pose):
        """
        Reset the camera pose

        Args:
            car_pose (Pose): Pose of car
        """
        with self.lock:
            self._reset(car_pose)
            self.is_reset_called = True

    def spawn_model(self, car_pose, camera_sdf_path):
        """
        Spawns a sdf model located in the given path

        Args:
            car_pose (Pose): Pose of car
            camera_sdf_path (string): full path to the location of sdf file
        """
        camera_sdf = self._get_sdf_string(camera_sdf_path)
        camera_pose = self._get_initial_camera_pose(car_pose)
        self.spawn_sdf_model(self.model_name, camera_sdf, self.model_name, camera_pose, "")

    def update_pose(self, car_pose, delta_time):
        """
        Update the camera pose

        Args:
            car_pose (Pose): Pose of car
            delta_time (float): time delta from last update
        """
        with self.lock:
            if self.is_reset_called:
                self._update(car_pose, delta_time)

    @abc.abstractmethod
    def _get_sdf_string(self, camera_sdf_path):
        """
        Reads the sdf file and converts it to a string in
        memory

        Args:
            camera_sdf_path (string): full path to the location of sdf file
        """
        raise NotImplementedError("Camera must read and convert model sdf file")

    @abc.abstractmethod
    def _reset(self, car_pose):
        """
        Reset the camera pose
        """
        raise NotImplementedError("Camera must be able to reset")

    @abc.abstractmethod
    def _update(self, car_pose, delta_time):
        """Update the camera pose"""
        raise NotImplementedError("Camera must be able to update")

    @abc.abstractmethod
    def _get_initial_camera_pose(self, car_pose):
        """compuate camera pose"""
        raise NotImplementedError("Camera must be able to compuate pose")

    def detach(self):
        """Detach camera from manager"""
        CameraManager.get_instance().remove(self, self.namespace)
