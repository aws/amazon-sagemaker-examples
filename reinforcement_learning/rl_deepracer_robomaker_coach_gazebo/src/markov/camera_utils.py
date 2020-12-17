import rospy
import time
import threading
from markov.cameras.camera_factory import CameraFactory
from markov.log_handler.deepracer_exceptions import GenericRolloutException
from markov.defaults import DEFAULT_MAIN_CAMERA, DEFAULT_SUB_CAMERA
from markov.gazebo_tracker.trackers.get_model_state_tracker import GetModelStateTracker

is_configure_camera_called = False
configure_camera_function_lock = threading.Lock()


# Amount of time (in seconds) to wait, in order to prevent model state from
# spamming logs while the model is loading
WAIT_TO_PREVENT_SPAM = 2


def wait_for_model(model_name, relative_entity_name):
    model = GetModelStateTracker.get_instance().get_model_state(model_name,
                                                                relative_entity_name,
                                                                blocking=True,
                                                                auto_sync=False)
    should_wait_for_model = not model.success
    while should_wait_for_model:
        time.sleep(WAIT_TO_PREVENT_SPAM)
        model = GetModelStateTracker.get_instance().get_model_state(model_name,
                                                                    relative_entity_name,
                                                                    blocking=True,
                                                                    auto_sync=False)
        should_wait_for_model = not model.success


def configure_camera(namespaces=None):
    namespaces = namespaces or ["racecar"]
    global is_configure_camera_called
    with configure_camera_function_lock:
        if not is_configure_camera_called:
            is_configure_camera_called = True
            main_camera_type = rospy.get_param("MAIN_CAMERA", DEFAULT_MAIN_CAMERA)
            sub_camera_type = rospy.get_param("SUB_CAMERA", DEFAULT_SUB_CAMERA)
            main_cameras = dict()
            for namespace in namespaces:
                wait_for_model(model_name=namespace, relative_entity_name="")
                main_cameras[namespace] = CameraFactory.create_instance(camera_type=main_camera_type,
                                                                        model_name="/{}/{}".format(namespace,
                                                                                                   "main_camera"),
                                                                        namespace=namespace)
            sub_camera = CameraFactory.create_instance(camera_type=sub_camera_type,
                                                       model_name="/{}".format("sub_camera"),
                                                       namespace=namespace)
            return main_cameras, sub_camera
        else:
            err_msg = "configure_camera called more than once. configure_camera MUST be called ONLY once!"
            raise GenericRolloutException(err_msg)

