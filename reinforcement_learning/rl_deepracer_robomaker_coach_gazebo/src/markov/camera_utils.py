import threading
import time

import rospy
from markov.cameras.camera_factory import CameraFactory
from markov.defaults import DEFAULT_MAIN_CAMERA, DEFAULT_SUB_CAMERA
from markov.gazebo_tracker.trackers.get_model_state_tracker import GetModelStateTracker
from markov.log_handler.deepracer_exceptions import GenericRolloutException

is_configure_camera_called = False
configure_camera_function_lock = threading.Lock()


# Amount of time (in seconds) to wait, in order to prevent model state from
# spamming logs while the model is loading
WAIT_TO_PREVENT_SPAM = 2


def wait_for_model(model_name, relative_entity_name):
    """wait for model to be ready and while not ready continuously waiting

    Args:
        model_name (str): wait model name
        relative_entity_name (str): relative entity name to model name
    """
    model = GetModelStateTracker.get_instance().get_model_state(
        model_name, relative_entity_name, blocking=True, auto_sync=False
    )
    should_wait_for_model = not model.success
    while should_wait_for_model:
        time.sleep(WAIT_TO_PREVENT_SPAM)
        model = GetModelStateTracker.get_instance().get_model_state(
            model_name, relative_entity_name, blocking=True, auto_sync=False
        )
        should_wait_for_model = not model.success


def configure_camera(namespaces=None, is_wait_for_model=True):
    """configure the top and follow car camear

    Args:
        namespaces (list): a list of all racecar namespace that top down and follow car
            cameras have to be configured
        is_wait_for_model (bool): boolean for whether wait for the racecar to be ready when
            configure to camera. The default value is True and we are waiting for racecar model
            for all besides virtual event because virtual event follow car camera does not follow
            a specific car at all time

    Returns:
        tuple(list(FollowCarCamera), TopCamera): tuple of a list of FollowCarCamera instance and
        a TopCamera instance
    """
    namespaces = namespaces or ["racecar"]
    global is_configure_camera_called
    with configure_camera_function_lock:
        if not is_configure_camera_called:
            is_configure_camera_called = True
            main_camera_type = rospy.get_param("MAIN_CAMERA", DEFAULT_MAIN_CAMERA)
            sub_camera_type = rospy.get_param("SUB_CAMERA", DEFAULT_SUB_CAMERA)
            main_cameras = dict()
            for namespace in namespaces:
                if is_wait_for_model:
                    wait_for_model(model_name=namespace, relative_entity_name="")
                main_cameras[namespace] = CameraFactory.create_instance(
                    camera_type=main_camera_type,
                    model_name="/{}/{}".format(namespace, "main_camera"),
                    namespace=namespace,
                )
            sub_camera = CameraFactory.create_instance(
                camera_type=sub_camera_type,
                model_name="/{}".format("sub_camera"),
                namespace=namespace,
            )
            return main_cameras, sub_camera
        else:
            err_msg = (
                "configure_camera called more than once. configure_camera MUST be called ONLY once!"
            )
            raise GenericRolloutException(err_msg)
