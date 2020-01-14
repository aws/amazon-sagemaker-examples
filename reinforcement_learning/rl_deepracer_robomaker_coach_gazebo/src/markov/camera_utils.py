import rospy
from markov.cameras.camera_factory import CameraFactory
from markov.defaults import DEFAULT_MAIN_CAMERA, DEFAULT_SUB_CAMERA


def configure_camera():
    main_camera = rospy.get_param("MAIN_CAMERA", DEFAULT_MAIN_CAMERA)
    sub_camera = rospy.get_param("SUB_CAMERA", DEFAULT_SUB_CAMERA)
    main_camera = CameraFactory.get_instance(main_camera)
    main_camera.topic_name = "main_camera"
    sub_camera = CameraFactory.get_instance(sub_camera)
    sub_camera.topic_name = "sub_camera"
    return main_camera, sub_camera
