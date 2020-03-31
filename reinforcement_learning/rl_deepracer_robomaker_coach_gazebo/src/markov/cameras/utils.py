import math
import numpy as np
from markov.deepracer_exceptions import GenericRolloutException
from markov.track_geom.utils import euler_to_quaternion, apply_orientation, inverse_quaternion


def get_angle_between_two_points_2d_rad(pt1, pt2):
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        return math.pi if dx == 0 else math.atan2(dy, dx)


def lerp(current, target, fraction):
    fraction = np.clip(fraction, 0.0, 1.0)
    return current + fraction * (target - current)


def lerp_angle_rad(current, target, fraction):
    while current > target + np.pi:
        target += np.pi * 2.0
    while target > current + np.pi:
        target -= np.pi * 2.0
    return lerp(current, target, fraction)


def normalize(vector):
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def project_to_2d(point_on_plane, plane_center, plane_width, plane_height, plane_quaternion):
    if not isinstance(point_on_plane, list) and not isinstance(point_on_plane, tuple) \
            and not isinstance(point_on_plane, np.ndarray):
        raise GenericRolloutException("point_on_plane must be a type of list, tuple, or numpy.ndarray")
    if not isinstance(plane_center, list) and not isinstance(plane_center, tuple) \
            and not isinstance(plane_center, np.ndarray):
        raise GenericRolloutException("plane_center must be a type of list, tuple, or numpy.ndarray")
    if not isinstance(plane_quaternion, list) and not isinstance(plane_quaternion, tuple) \
            and not isinstance(plane_quaternion, np.ndarray):
        raise GenericRolloutException("plane_quaternion must be a type of list, tuple, or numpy.ndarray")

    point_on_plane = np.array(point_on_plane)
    plane_center = np.array(plane_center)
    plane_quaternion = np.array(plane_quaternion)

    # Transpose the center back to origin
    point_on_plane_from_origin = point_on_plane - plane_center

    # Reverse the rotation so plane can align back to y-axis
    inverse_cam_quaternion = inverse_quaternion(plane_quaternion)
    point_on_y_axis = apply_orientation(inverse_cam_quaternion, point_on_plane_from_origin)

    # Rotate pitch 90 degree and yaw 90 degree, so plane will align to x and y axis
    # Remember rotation order is roll, pitch, yaw in euler_to_quaternion method
    project_2d_quaternion = euler_to_quaternion(pitch=np.pi / 2.0, yaw=np.pi / 2.0)
    point_on_2d_plane = apply_orientation(project_2d_quaternion, point_on_y_axis)

    # Align plane to origin at x, y = (0, 0)
    point_on_2d_plane = point_on_2d_plane + np.array([plane_width / 2.0, plane_height / 2.0, 0.0])

    # Re-scale x and y space between 0 and 1
    return (point_on_2d_plane[0] / plane_width), (point_on_2d_plane[1] / plane_height)


def ray_plane_intersect(ray_origin, ray_dir, plane_normal, plane_offset, threshold=0.0001):
    ray_dir = normalize(ray_dir) if np.dot(ray_dir, ray_dir) - 1.0 > threshold else ray_dir
    plane_normal = normalize(plane_normal) if np.dot(plane_normal, plane_normal) - 1.0 > threshold else plane_normal
    point_on_plane = None
    denominator = np.dot(plane_normal, ray_dir)
    if np.abs(denominator) >= threshold:
        t = (plane_offset - np.dot(ray_origin, plane_normal)) / denominator
        # Point on near plane that intersects the ray and the plane
        point_on_plane = ray_origin + t * ray_dir
    return point_on_plane