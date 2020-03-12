import rospy
import threading
from markov.cameras.camera_factory import CameraFactory
from markov.deepracer_exceptions import GenericRolloutException
from markov.defaults import DEFAULT_MAIN_CAMERA, DEFAULT_SUB_CAMERA


is_configure_camera_called = False
configure_camera_function_lock = threading.Lock()


def configure_camera(namespaces=["racecar"]):
    global is_configure_camera_called
    with configure_camera_function_lock:
        if not is_configure_camera_called:
            is_configure_camera_called = True
            main_camera_type = rospy.get_param("MAIN_CAMERA", DEFAULT_MAIN_CAMERA)
            sub_camera_type = rospy.get_param("SUB_CAMERA", DEFAULT_SUB_CAMERA)
            main_camera = dict()
            for namespace in namespaces:
                main_camera[namespace] = CameraFactory.create_instance(camera_type=main_camera_type,
                                                            topic_name="/{}/{}".format(namespace,"main_camera"),
                                                            namespace=namespace)
            sub_camera = CameraFactory.create_instance(camera_type=sub_camera_type,
                                                    topic_name="/{}".format("sub_camera"),
                                                    namespace=namespace)
            return main_camera, sub_camera
        else:
            err_msg = "configure_camera called more than once. configure_camera MUST be called ONLY once!"
            raise GenericRolloutException(err_msg)

