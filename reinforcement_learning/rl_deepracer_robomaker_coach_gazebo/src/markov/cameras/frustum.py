import numpy as np
import math
import threading
from markov.deepracer_exceptions import GenericRolloutException
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.architecture.constants import Input
from markov.track_geom.utils import euler_to_quaternion, quaternion_to_euler, apply_orientation
from markov.cameras.constants import GazeboWorld
from markov.cameras.utils import normalize, project_to_2d, ray_plane_intersect

import rospy
from gazebo_msgs.srv import GetModelState


class Frustum(object):
    def __init__(self, agent_name, observation_list):
        self.agent_name = agent_name

        # define inward direction
        self.ccw = True

        rospy.wait_for_service('/gazebo/get_model_state')
        self.model_state_client = ServiceProxyWrapper('/gazebo/get_model_state', GetModelState)
        self.frustums = []
        self.cam_poses = []
        self.near_plane_infos = []
        self.threshold = 0.0001
        self.lock = threading.Lock()

        # TODO: Hard coding the frustum values for now. These values need to be loaded from SDF directly
        self.near = 0.2
        self.far = 300.0
        self.horizontal_fov = 1.13  # rad ~ 65 degree
        self.view_ratio = 480.0 / 640.0

        # TODO: Remove below camera_offsets and camera_pitch variables if we can get Camera pose directly
        self.camera_offsets = []
        if Input.STEREO.value in observation_list:
            # Hard coded Stereo Camera Offset Position from basis_link
            self.camera_offsets.append(np.array([0.2, -0.03, 0.164]))
            self.camera_offsets.append(np.array([0.2, 0.03, 0.164]))
        else:
            # Hard coded Front Facing Camera Offset Position from basis_link
            self.camera_offsets.append(np.array([0.2, 0, 0.164]))
        self.camera_pitch = 0.261799

    @staticmethod
    def get_forward_vec(quaternion):
        return apply_orientation(quaternion, GazeboWorld.forward)

    @staticmethod
    def get_up_vec(quaternion):
        return apply_orientation(quaternion, GazeboWorld.up)

    @staticmethod
    def get_right_vec(quaternion):
        return apply_orientation(quaternion, GazeboWorld.right)

    def update(self):
        """
        Update the frustums
        """
        with self.lock:
            self.frustums = []
            self.cam_poses = []
            self.near_plane_infos = []
            # Retrieve car pose to calculate camera pose.
            car_model_state = self.model_state_client(self.agent_name, '')
            car_pos = np.array([car_model_state.pose.position.x, car_model_state.pose.position.y,
                                car_model_state.pose.position.z])
            car_quaternion = [car_model_state.pose.orientation.x,
                              car_model_state.pose.orientation.y,
                              car_model_state.pose.orientation.z,
                              car_model_state.pose.orientation.w]
            for camera_offset in self.camera_offsets:
                # Get camera position by applying position offset from the car position.
                cam_pos = car_pos + apply_orientation(car_quaternion, camera_offset)
                # Get camera rotation by applying car rotation and pitch angle of camera.
                _, _, yaw = quaternion_to_euler(x=car_quaternion[0],
                                                y=car_quaternion[1],
                                                z=car_quaternion[2],
                                                w=car_quaternion[3])
                cam_quaternion = np.array(euler_to_quaternion(pitch=self.camera_pitch, yaw=yaw))
                # Calculate frustum with camera position and rotation.
                planes, cam_pose, near_plane_info = self._calculate_frustum_planes(cam_pos=cam_pos,
                                                                                   cam_quaternion=cam_quaternion)
                self.frustums.append(planes)
                self.cam_poses.append(cam_pose)
                self.near_plane_infos.append(near_plane_info)

    def _calculate_frustum_planes(self, cam_pos, cam_quaternion):
        cam_pos = np.array(cam_pos)
        cam_quaternion = np.array(cam_quaternion)
        cam_forward = Frustum.get_forward_vec(cam_quaternion)
        cam_up = Frustum.get_up_vec(cam_quaternion)
        cam_right = Frustum.get_right_vec(cam_quaternion)

        near_center = cam_pos + cam_forward * self.near
        far_center = cam_pos + cam_forward * self.far

        near_width = 2.0 * math.tan(self.horizontal_fov * 0.5) * self.near
        near_height = near_width * self.view_ratio
        far_width = 2.0 * math.tan(self.horizontal_fov * 0.5) * self.far
        far_height = far_width * self.view_ratio

        far_top_left = far_center + cam_up * (far_height * 0.5) - cam_right * (far_width * 0.5)
        far_top_right = far_center + cam_up * (far_height * 0.5) + cam_right * (far_width * 0.5)
        far_bottom_left = far_center - cam_up * (far_height * 0.5) - cam_right * (far_width * 0.5)
        far_bottom_right = far_center - cam_up * (far_height * 0.5) + cam_right * (far_width * 0.5)

        near_top_left = near_center + cam_up * (near_height * 0.5) - cam_right * (near_width * 0.5)
        near_top_right = near_center + cam_up * (near_height * 0.5) + cam_right * (near_width * 0.5)
        near_bottom_left = near_center - cam_up * (near_height * 0.5) - cam_right * (near_width * 0.5)
        near_bottom_right = near_center - cam_up * (near_height * 0.5) + cam_right * (near_width * 0.5)
        planes = []

        # near plane
        if self.ccw:
            p0, p1, p2 = near_bottom_right, near_bottom_left, near_top_left
        else:
            p0, p1, p2 = near_top_left, near_bottom_left, near_bottom_right
        near_plane_normal = normalize(np.cross(p1 - p0, p2 - p1))
        near_plane_offset = np.dot(near_plane_normal, p0)
        planes.append((near_plane_normal, near_plane_offset))

        # far plane
        if self.ccw:
            p0, p1, p2 = far_bottom_right, far_top_right, far_top_left
        else:
            p0, p1, p2 = far_top_left, far_top_right, far_bottom_right
        far_plane_normal = normalize(np.cross(p1 - p0, p2 - p1))
        far_plane_offset = np.dot(far_plane_normal, p0)
        planes.append((far_plane_normal, far_plane_offset))

        # left plane
        if self.ccw:
            p0, p1, p2 = near_bottom_left, far_bottom_left, far_top_left
        else:
            p0, p1, p2 = far_top_left, far_bottom_left, near_bottom_left
        left_plane_normal = normalize(np.cross(p1 - p0, p2-p1))
        left_plane_offset = np.dot(left_plane_normal, p0)
        planes.append((left_plane_normal, left_plane_offset))

        # right plane
        if self.ccw:
            p0, p1, p2 = near_top_right, far_top_right, far_bottom_right
        else:
            p0, p1, p2 = far_bottom_right, far_top_right, near_top_right
        right_plane_normal = normalize(np.cross(p1 - p0, p2-p1))
        right_plane_offset = np.dot(right_plane_normal, p0)
        planes.append((right_plane_normal, right_plane_offset))

        # top plane
        if self.ccw:
            p0, p1, p2 = near_top_left, far_top_left, far_top_right
        else:
            p0, p1, p2 = far_top_right, far_top_left, near_top_left
        top_plane_normal = normalize(np.cross(p1 - p0, p2-p1))
        top_plane_offset = np.dot(top_plane_normal, p0)
        planes.append((top_plane_normal, top_plane_offset))

        # bottom plane
        if self.ccw:
            p0, p1, p2 = near_bottom_right, far_bottom_right, far_bottom_left
        else:
            p0, p1, p2 = far_bottom_left, far_bottom_right, near_bottom_right
        bottom_plane_normal = normalize(np.cross(p1 - p0, p2 - p1))
        bottom_plane_offset = np.dot(bottom_plane_normal, p0)
        planes.append((bottom_plane_normal, bottom_plane_offset))

        cam_pose = {
            "position": cam_pos,
            "orientation": cam_quaternion
        }

        near_plane_info = {
            "width": near_width,
            "height": near_height,
            "position": near_center,
            "bounding_box": {
                "bottom_left": near_bottom_left,
                "top_left": near_top_left,
                "top_right": near_top_right,
                "bottom_right": near_bottom_right,
            },
            "normal": near_plane_normal,
            "offset": near_plane_offset
        }

        return planes, cam_pose, near_plane_info

    def is_visible(self, point):
        """
        Returns True if point is in any frustums defined
        - If point is in any frustum then the point is considered as visible
        point - 3d position
        """
        return any(self.test_visibility(point))

    def test_visibility(self, point):
        """
        Test the visibility of the point for each frustums defined.
        (True indicates the point is in the frustum otherwise False)
        Returns list of test results where the list size is same as number of frustums defined.
        - This will be useful when multiple frustums are defined (like Stereo Camera) and want to find
          which lens that object is visible.
        point - 3d position
        """
        with self.lock:
            if not isinstance(point, list) and not isinstance(point, tuple) \
                    and not isinstance(point, np.ndarray):
                raise GenericRolloutException("point must be a type of list, tuple, or numpy.ndarray")
            target_pos = np.array(point)
            frustum_tests = []
            for frustum_planes in self.frustums:
                is_in_frustum = True
                for plane in frustum_planes:
                    plane_normal, plane_offset = plane
                    if np.dot(plane_normal, target_pos) - plane_offset <= 0.0:
                        is_in_frustum = False
                        break
                frustum_tests.append(is_in_frustum)
            return frustum_tests

    def to_viewport_point(self, point):
        with self.lock:
            if not isinstance(point, list) and not isinstance(point, tuple) \
                    and not isinstance(point, np.ndarray):
                raise GenericRolloutException("point must be a type of list, tuple, or numpy.ndarray")
            ray_origin = np.array(point)
            points_in_viewports = []
            for cam_pose, near_plane_info in zip(self.cam_poses, self.near_plane_infos):
                cam_pos = cam_pose["position"]
                cam_quaternion = cam_pose["orientation"]

                near_width = near_plane_info["width"]
                near_height = near_plane_info["height"]
                near_center = near_plane_info["position"]
                near_normal = near_plane_info["normal"]
                near_offset = near_plane_info["offset"]

                ray_dir = normalize(cam_pos - ray_origin)

                point_on_plane = ray_plane_intersect(ray_origin=ray_origin,
                                                     ray_dir=ray_dir,
                                                     plane_normal=near_normal,
                                                     plane_offset=near_offset,
                                                     threshold=self.threshold)
                if point_on_plane is None:
                    points_in_viewports.append((-1.0, -1.0))
                else:
                    point_in_viewport = project_to_2d(point_on_plane=point_on_plane,
                                                      plane_center=near_center,
                                                      plane_width=near_width,
                                                      plane_height=near_height,
                                                      plane_quaternion=cam_quaternion)
                    points_in_viewports.append(point_in_viewport)
            return points_in_viewports
