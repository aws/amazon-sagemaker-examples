import threading
from markov.deepracer_exceptions import GenericRolloutException


class CameraManager(object):
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
        self.lock = threading.Lock()
        self.camera_namespaces = {'default': set()}

        # there should be only one camera manager instance
        CameraManager._instance_ = self

    def add(self, camera, namespace=None):
        """Add a camera to manage

        Args:
            camera (object): Camera object
            namespace (str): namespace
        """
        if namespace == '*':
            raise GenericRolloutException("* is not allowed to use as namespace name.")
        with self.lock:
            namespace = namespace or 'default'
            if namespace not in self.camera_namespaces:
                self.camera_namespaces[namespace] = set()
            self.camera_namespaces[namespace].add(camera)

    def remove(self, camera, namespace='*'):
        """Remove the camera in given namespace from manager.
        - namespace '*' will try to remove given camera from every namespaces

        Args:
            camera (object): Camera object
            namespace (str): namespace
        """
        with self.lock:
            if namespace == '*':
                for cur_namespace in self.camera_namespaces:
                    self.camera_namespaces[cur_namespace].remove(camera)
            else:
                self.camera_namespaces[namespace].remove(camera)

    def reset(self, state, namespace='*'):
        """Reset camera pose on given namespace
        - namespace '*' will reset camera in every namespaces

        Args:
            state (object): state object for reset
            namespace (str): namespace
        """
        with self.lock:
            if namespace == '*':
                for cur_namespace in self.camera_namespaces:
                    for camera in self.camera_namespaces[cur_namespace]:
                        camera.reset_pose(state)
            else:
                for camera in self.camera_namespaces[namespace]:
                    camera.reset_pose(state)

    def update(self, state, delta_time, namespace='*'):
        """Update camera with state and delta_time
        - namespace '*' will update camera in every namespaces
        """
        with self.lock:
            if namespace == '*':
                for cur_namespace in self.camera_namespaces:
                    for camera in self.camera_namespaces[cur_namespace]:
                        camera.update_pose(state, delta_time)
            else:
                for camera in self.camera_namespaces[namespace]:
                    camera.update_pose(state, delta_time)
