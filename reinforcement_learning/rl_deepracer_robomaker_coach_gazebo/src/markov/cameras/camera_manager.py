import threading
from markov.cameras.camera_factory import CameraFactory


class CameraManager(object):
    """
    Camera Manager class that manages multiple camera
    """
    def __init__(self, camera_types):
        self.lock = threading.Lock()
        self.camera_types = set(camera_types) or set()

    def add(self, camera_type):
        """Add the name of new camera to manage"""
        with self.lock:
            self.camera_types.add(camera_type)

    def remove(self, camera_type):
        """Remove the camera from manager"""
        with self.lock:
            self.camera_types.remove(camera_type)

    def reset(self, state):
        """Reset camera pose

        Only cameras that are instantiated will get reset
        """
        with self.lock:
            for camera_type in self.camera_types:
                camera_class = CameraFactory.get_class(camera_type)
                if camera_class.has_instance():
                    camera = CameraFactory.get_instance(camera_type)
                    camera.reset_pose(state)

    def update(self, state, delta_time):
        """Update camera with state and delta_time

        Only cameras that are instantiated will get reset
        """
        with self.lock:
            for camera_type in self.camera_types:
                camera_class = CameraFactory.get_class(camera_type)
                if camera_class.has_instance():
                    camera = CameraFactory.get_instance(camera_type)
                    camera.update_pose(state, delta_time)
