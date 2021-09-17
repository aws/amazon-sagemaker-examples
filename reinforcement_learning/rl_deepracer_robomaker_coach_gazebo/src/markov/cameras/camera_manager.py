import copy
import logging
import threading

from markov.gazebo_tracker.abs_tracker import AbstractTracker
from markov.gazebo_tracker.trackers.get_model_state_tracker import GetModelStateTracker
from markov.log_handler.deepracer_exceptions import GenericRolloutException
from markov.log_handler.logger import Logger

LOG = Logger(__name__, logging.INFO).get_logger()


class CameraManager(AbstractTracker):
    """
    Camera Manager class that manages multiple camera
    """

    _instance_ = None

    @staticmethod
    def get_instance():
        """Method for getting a reference to the camera manager object"""
        if CameraManager._instance_ is None:
            CameraManager()
        return CameraManager._instance_

    def __init__(self):
        if CameraManager._instance_ is not None:
            raise GenericRolloutException("Attempting to construct multiple Camera Manager")
        self.lock = threading.RLock()
        self.camera_namespaces = {}

        # there should be only one camera manager instance
        CameraManager._instance_ = self
        super(CameraManager, self).__init__()

    def pop(self, namespace):
        """pop camera manager namespace

        Args:
            namespace(str): camera manager namespace string

        Returns:
            set: sets of cameras under a specific namespace. Otherwise, None
        """
        with self.lock:
            cameras_set = self.camera_namespaces.pop(namespace, None)
        LOG.info("[CameraManager]: Popped camera_namespaces %s", namespace)
        return cameras_set

    def add(self, camera, namespace):
        """Add a camera to manage

        Args:
            camera (object): Camera object
            namespace (str): namespace
        """
        if not namespace or namespace == "*":
            raise GenericRolloutException("namespace must be provided and '*' is not allowed.")
        with self.lock:
            if namespace not in self.camera_namespaces:
                self.camera_namespaces[namespace] = set()
            self.camera_namespaces[namespace].add(camera)
        LOG.info("[CameraManager]: Added %s to camera_namespaces %s", camera, namespace)

    def remove(self, camera, namespace="*"):
        """Remove the camera in given namespace from manager.
        - namespace '*' will try to remove given camera from every namespaces

        Args:
            camera (object): Camera object
            namespace (str): namespace
        """
        with self.lock:
            if namespace == "*":
                for cur_namespace in self.camera_namespaces:
                    self.camera_namespaces[cur_namespace].discard(camera)
            else:
                self.camera_namespaces[namespace].discard(camera)
        LOG.info("[CameraManager]: Removed %s from camera_namespaces %s", camera, namespace)

    def reset(self, car_pose, namespace="*"):
        """Reset camera pose on given namespace
        - namespace '*' will reset camera in every namespaces

        Args:
            car_pose (Pose): Pose of car
            namespace (str): namespace
        """
        with self.lock:
            if namespace == "*":
                for cur_namespace in self.camera_namespaces:
                    for camera in self.camera_namespaces[cur_namespace]:
                        camera.reset_pose(car_pose)
            else:
                for camera in self.camera_namespaces[namespace]:
                    camera.reset_pose(car_pose)

    def update_tracker(self, delta_time, sim_time):
        """
        Callback when sim time is updated

        Args:
            delta_time (float): time diff from last call
            sim_time (Clock): simulation time
        """
        with self.lock:
            camera_name_space = copy.copy(self.camera_namespaces)
            for namespace in camera_name_space:
                car_model_state = GetModelStateTracker.get_instance().get_model_state(namespace, "")
                self._update(
                    car_pose=car_model_state.pose, delta_time=delta_time, namespace=namespace
                )

    def _update(self, car_pose, delta_time, namespace="*"):
        """Update camera with state and delta_time
        - namespace '*' will update camera in every namespaces
        """
        if namespace == "*":
            for cur_namespace in self.camera_namespaces:
                for camera in self.camera_namespaces[cur_namespace]:
                    camera.update_pose(car_pose, delta_time)
        else:
            for camera in self.camera_namespaces[namespace]:
                camera.update_pose(car_pose, delta_time)
