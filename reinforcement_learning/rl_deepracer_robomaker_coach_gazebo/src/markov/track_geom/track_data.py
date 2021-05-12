"""This module is used to manage the track related data"""
import math
import os
import threading
from collections import OrderedDict, deque
from enum import Enum, unique

import numpy as np
import rospkg
import rospy
from geometry_msgs.msg import Pose
from markov import utils
from markov.agent_ctrl.constants import RewardParam
from markov.cameras.frustum_manager import FrustumManager
from markov.log_handler.deepracer_exceptions import GenericRolloutException
from markov.track_geom.constants import ParkLocation, TrackNearDist, TrackNearPnts
from markov.track_geom.utils import (
    apply_orientation,
    euler_to_quaternion,
    find_prev_next,
    quaternion_to_euler,
)
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import LinearRing, LineString


@unique
class FiniteDifference(Enum):
    CENTRAL_DIFFERENCE = 1
    FORWARD_DIFFERENCE = 2


class TrackLine(object):
    def __init__(self, line):
        self.line = line
        self.ndists = [
            self.line.project(Point(p), normalized=True) for p in self.line.coords[:-1]
        ] + [1.0]

    def __getattr__(self, name):
        return getattr(self.line, name)

    def find_prev_next_waypoints(self, distance, normalized=False):
        ndist = distance if normalized else distance / self.line.length
        return find_prev_next(self.ndists, ndist)

    def interpolate_yaw(
        self,
        distance,
        normalized=False,
        position=None,
        finite_difference=FiniteDifference.CENTRAL_DIFFERENCE,
    ):
        prev_index, next_index = self.find_prev_next_waypoints(distance, normalized)
        if finite_difference == FiniteDifference.CENTRAL_DIFFERENCE:
            yaw = math.atan2(
                self.line.coords[next_index][1] - self.line.coords[prev_index][1],
                self.line.coords[next_index][0] - self.line.coords[prev_index][0],
            )
        elif finite_difference == FiniteDifference.FORWARD_DIFFERENCE:
            if not position:
                position = self.interpolate(distance, normalized)
            yaw = math.atan2(
                self.line.coords[next_index][1] - position.y,
                self.line.coords[next_index][0] - position.x,
            )
        else:
            raise ValueError("Unrecognized FiniteDifference enum value")
        return yaw

    def interpolate_pose(
        self, distance, normalized=False, finite_difference=FiniteDifference.CENTRAL_DIFFERENCE
    ):
        pose = Pose()
        position = self.interpolate(distance, normalized)
        yaw = self.interpolate_yaw(distance, normalized, position, finite_difference)
        orientation = euler_to_quaternion(yaw=yaw)
        pose.position.x = position.x
        pose.position.y = position.y
        pose.position.z = 0.0
        pose.orientation.x = orientation[0]
        pose.orientation.y = orientation[1]
        pose.orientation.z = orientation[2]
        pose.orientation.w = orientation[3]
        return pose


class TrackData(object):
    """This class is responsible for managing all the track geometry, the object should
    be created and shared between agents on the track
    """

    # The track data will be a singelton to prevent copying across multiple agents
    _instance_ = None

    @staticmethod
    def get_instance():
        """Method for geting a reference to the track data object"""
        if TrackData._instance_ is None:
            TrackData()
        return TrackData._instance_

    @property
    def center_line(self):
        """center line property depending on direction"""
        if self._reverse_dir_:
            return self._center_line_reverse_
        return self._center_line_forward_

    @property
    def inner_border(self):
        """inner board property depending on direction"""
        if self._reverse_dir_:
            return self._inner_border_reverse_
        return self._inner_border_forward_

    @property
    def outer_border(self):
        """outer board property depending on direction"""
        if self._reverse_dir_:
            return self._outer_border_reverse_
        return self._outer_border_forward_

    @property
    def inner_lane(self):
        """inner lane property depending on direction"""
        if self._reverse_dir_:
            return self._inner_lane_reverse_
        return self._inner_lane_forward_

    @property
    def outer_lane(self):
        """outer lane property depending on direction"""
        if self._reverse_dir_:
            return self._outer_lane_reverse_
        return self._outer_lane_forward_

    @property
    def reverse_dir(self):
        """reverse direction getter"""
        return self._reverse_dir_

    @property
    def is_ccw(self):
        """ccw direction getter"""
        return self._is_ccw_ ^ self._reverse_dir_

    @reverse_dir.setter
    def reverse_dir(self, val):
        """reverse direction setter"""
        self._reverse_dir_ = val

    @property
    def park_positions(self):
        """park positions getter"""
        return self._park_positions_

    @park_positions.setter
    def park_positions(self, val):
        """park positions setter"""
        self._park_positions_ = deque(val)

    def pop_park_position(self):
        """pop first available park position"""
        return self._park_positions_.popleft()

    @property
    def park_location(self):
        """Park location getter"""
        return self._park_location

    def __init__(self):
        """Instantiates the class and creates clients for the relevant ROS services"""
        self._park_positions_ = deque()
        self._park_location = ParkLocation(
            rospy.get_param("PARK_LOCATION", ParkLocation.BOTTOM.value).lower()
        )
        self._reverse_dir_ = utils.str2bool(rospy.get_param("REVERSE_DIR", False))
        if TrackData._instance_ is not None:
            raise GenericRolloutException("Attempting to construct multiple TrackData objects")
        try:
            rospack = rospkg.RosPack()
            deepracer_path = rospack.get_path("deepracer_simulation_environment")
            waypoints_path = os.path.join(
                deepracer_path, "routes", "{}.npy".format(rospy.get_param("WORLD_NAME"))
            )
            self._is_bot_car_ = int(rospy.get_param("NUMBER_OF_BOT_CARS", 0)) > 0
            self._bot_car_speed_ = float(rospy.get_param("BOT_CAR_SPEED", 0.0))
            waypoints = np.load(waypoints_path)

            self.is_loop = np.all(waypoints[0, :] == waypoints[-1, :])
            poly_func = LinearRing if self.is_loop else LineString
            # forward direction
            self._center_line_forward_ = TrackLine(poly_func(waypoints[:, 0:2]))
            self._inner_border_forward_ = TrackLine(poly_func(waypoints[:, 2:4]))
            self._outer_border_forward_ = TrackLine(poly_func(waypoints[:, 4:6]))
            self._inner_lane_forward_ = TrackLine(
                poly_func((waypoints[:, 2:4] + waypoints[:, 0:2]) / 2)
            )
            self._outer_lane_forward_ = TrackLine(
                poly_func((waypoints[:, 4:6] + waypoints[:, 0:2]) / 2)
            )
            # reversed direction
            self._center_line_reverse_ = TrackLine(poly_func(waypoints[:, 0:2][::-1]))
            self._inner_border_reverse_ = TrackLine(poly_func(waypoints[:, 2:4][::-1]))
            self._outer_border_reverse_ = TrackLine(poly_func(waypoints[:, 4:6][::-1]))
            self._inner_lane_reverse_ = TrackLine(
                poly_func((waypoints[:, 2:4][::-1] + waypoints[:, 0:2][::-1]) / 2)
            )
            self._outer_lane_reverse_ = TrackLine(
                poly_func((waypoints[:, 4:6][::-1] + waypoints[:, 0:2][::-1]) / 2)
            )
            if self.is_loop:
                self._inner_poly_ = Polygon(self.center_line, [self.inner_border])
                self._road_poly_ = Polygon(self.outer_border, [self.inner_border])
                self._is_ccw_ = self._center_line_forward_.is_ccw
            else:
                self._inner_poly_ = Polygon(
                    np.vstack((self.center_line.line, np.flipud(self.inner_border)))
                )
                self._road_poly_ = Polygon(
                    np.vstack((self.outer_border, np.flipud(self.inner_border)))
                )
                self._is_ccw_ = True

            self.object_poses = OrderedDict()
            self.object_dims = OrderedDict()
            self.noncollidable_objects = set()
            self.noncollidable_object_lock = threading.Lock()

            # There should only be one track data object
            TrackData._instance_ = self
            # declare a lock to prevent read and write at the same time
            self._lock_ = threading.Lock()

        except Exception as ex:
            raise GenericRolloutException("Failed to create track data: {}".format(ex))

    def initialize_object(self, name, initial_pose, object_dimensions):
        self.object_poses[name] = initial_pose
        self.object_dims[name] = object_dimensions

    def remove_object(self, name):
        """remove object from object_poses and object_dims member variable

        Args:
            name (str): object name to remove
        """
        self.object_poses.pop(name, None)
        self.object_dims.pop(name, None)

    def update_object_pose(self, name, object_pose):
        if name in self.object_poses:
            self.object_poses[name] = object_pose
        else:
            raise GenericRolloutException("Failed to update object (unrecognized): {}".format(name))

    def get_object_pose(self, name):
        return self.object_poses[name]

    def find_prev_next_waypoints(self, distance, normalized=False):
        return self.center_line.find_prev_next_waypoints(distance, normalized)

    @staticmethod
    def get_nearest_dist(near_pnt_dict, model_point):
        """Auxiliary method for computing the distance to the nearest points given in
        near_pnt_dict.
        near_pnt_dict - Dictionary containing the keys specified in TrackNearPnts
        model_point - Object returned by calling the model state service which contains
                      the position data for the agent
        """
        try:
            dist_from_cent = near_pnt_dict[TrackNearPnts.NEAR_PNT_CENT.value].distance(model_point)
            dist_from_in = near_pnt_dict[TrackNearPnts.NEAR_PNT_IN.value].distance(model_point)
            dist_from_out = near_pnt_dict[TrackNearPnts.NEAR_PNT_OUT.value].distance(model_point)

            return {
                TrackNearDist.NEAR_DIST_CENT.value: dist_from_cent,
                TrackNearDist.NEAR_DIST_IN.value: dist_from_in,
                TrackNearDist.NEAR_DIST_OUT.value: dist_from_out,
            }
        except Exception as ex:
            raise GenericRolloutException("Unable to compute nearest distance: {}".format(ex))

    def get_track_length(self):
        """Returns the length of the track"""
        try:
            return self.center_line.length
        except Exception as ex:
            raise GenericRolloutException("Unable to get track lenght: {}".format(ex))

    def get_way_pnts(self):
        """Returns a list containing all the way points"""
        try:
            return list(self.center_line.coords)
        except Exception as ex:
            raise GenericRolloutException("Unable to get way points: {}".format(ex))

    def get_norm_dist(self, model_point):
        """Returns the normalized position of the agent relative to the track
        model_point - Object returned by calling the model state service which contains
                      the position data for the agent
        """
        try:
            return self.center_line.project(model_point, normalized=True)
        except Exception as ex:
            raise GenericRolloutException("Unable to get norm dist: {}".format(ex))

    def get_nearest_points(self, model_point):
        """Returns a dictionary with the keys specified in TrackNearPnts containing
        the nearest way points to the agent.
        model_point - Object returned by calling the model state service which contains
                      the position data for the agent
        """
        try:
            near_pnt_ctr = self.center_line.interpolate(
                self.get_norm_dist(model_point), normalized=True
            )
            near_pnt_in = self.inner_border.interpolate(self.inner_border.project(near_pnt_ctr))
            near_pnt_out = self.outer_border.interpolate(self.outer_border.project(near_pnt_ctr))

            return {
                TrackNearPnts.NEAR_PNT_CENT.value: near_pnt_ctr,
                TrackNearPnts.NEAR_PNT_IN.value: near_pnt_in,
                TrackNearPnts.NEAR_PNT_OUT.value: near_pnt_out,
            }
        except Exception as ex:
            raise GenericRolloutException("Unable to get nearest points: {}".format(ex))

    def get_object_reward_params(self, racecar_name, model_point, car_pose):
        """Returns a dictionary with object-related reward function params."""
        with self._lock_:
            try:
                object_locations = [
                    [pose.position.x, pose.position.y]
                    for name, pose in self.object_poses.items()
                    if racecar_name not in name
                ]
                object_poses = [
                    pose for name, pose in self.object_poses.items() if racecar_name not in name
                ]
                if not object_locations:
                    return {}

                # Sort the object locations based on projected distance
                num_objects = len(object_locations)
                object_pdists = [self.center_line.project(Point(p)) for p in object_locations]
                object_headings = [0.0] * num_objects
                object_speeds = [0.0] * num_objects
                if self._is_bot_car_:
                    for i, object_pose in enumerate(object_poses):
                        _, _, yaw = quaternion_to_euler(
                            x=object_pose.orientation.x,
                            y=object_pose.orientation.y,
                            z=object_pose.orientation.z,
                            w=object_pose.orientation.w,
                        )
                        object_headings[i] = yaw
                        object_speeds[i] = self._bot_car_speed_

                # Find the prev/next objects
                model_pdist = self.center_line.project(model_point)
                object_order = np.argsort(object_pdists)
                object_pdists_ordered = [object_pdists[i] for i in object_order]
                prev_object_index, next_object_index = find_prev_next(
                    object_pdists_ordered, model_pdist
                )
                prev_object_index = object_order[prev_object_index]
                next_object_index = object_order[next_object_index]

                # Figure out which one is the closest
                object_points = [
                    Point([object_location[0], object_location[1]])
                    for object_location in object_locations
                ]
                prev_object_point = object_points[prev_object_index]
                next_object_point = object_points[next_object_index]
                prev_object_dist = model_point.distance(prev_object_point)
                next_object_dist = model_point.distance(next_object_point)
                if prev_object_dist < next_object_dist:
                    closest_object_point = prev_object_point
                else:
                    closest_object_point = next_object_point

                # Figure out whether objects is left of center based on direction
                objects_left_of_center = [
                    self._inner_poly_.contains(p) ^ (not self.is_ccw) for p in object_points
                ]

                # Get object distances to centerline
                objects_distance_to_center = [self.center_line.distance(p) for p in object_points]

                # Figure out if the next object is in the camera view
                objects_in_camera = self.get_objects_in_camera_frustums(
                    agent_name=racecar_name, car_pose=car_pose
                )
                is_next_object_in_camera = any(
                    object_in_camera_idx == next_object_index
                    for object_in_camera_idx, _ in objects_in_camera
                )

                # Determine if they are in the same lane
                return {
                    RewardParam.CLOSEST_OBJECTS.value[0]: [prev_object_index, next_object_index],
                    RewardParam.OBJECT_LOCATIONS.value[0]: object_locations,
                    RewardParam.OBJECTS_LEFT_OF_CENTER.value[0]: objects_left_of_center,
                    RewardParam.OBJECT_SPEEDS.value[0]: object_speeds,
                    RewardParam.OBJECT_HEADINGS.value[0]: object_headings,
                    RewardParam.OBJECT_CENTER_DISTS.value[0]: objects_distance_to_center,
                    RewardParam.OBJECT_CENTERLINE_PROJECTION_DISTANCES.value[0]: object_pdists,
                    RewardParam.OBJECT_IN_CAMERA.value[0]: is_next_object_in_camera,
                }
            except Exception as ex:
                raise GenericRolloutException("Unable to get object reward params: {}".format(ex))

    def get_distance_from_next_and_prev(self, model_point, prev_index, next_index):
        """Returns a tuple, where the first value is the distance to the given previous points
        and the second value is the distance to given next point.
        model_point - Object returned by calling the model state service which contains
                      the position data for the agent
        prev_index - Integer representing the index of the previous point
        next_index - Integer representing the index of the next point
        """
        try:
            dist_from_prev = model_point.distance(Point(self.center_line.coords[prev_index]))
            dist_from_next = model_point.distance(Point(self.center_line.coords[next_index]))
            return dist_from_prev, dist_from_next
        except Exception as ex:
            raise GenericRolloutException(
                "Unable to get distance to prev and next points: {}".format(ex)
            )

    def points_on_track(self, points):
        """Returns a boolean list, where entries of true represent a point in the points list
        being on the track, and values of false represent a point in the points list being
        of the track.
        points - List of points who will be checked for being on or off the track
        """
        try:
            return [self._road_poly_.contains(pnt) for pnt in points]
        except Exception as ex:
            raise GenericRolloutException("Unable to get points on track {}".format(ex))

    @staticmethod
    def get_object_bounding_rect(object_pose, object_dims):
        """
        Returns a list of points (numpy.ndarray) of bounding rectangle on the floor.
        object_pose - Object pose object.
        object_dims - Tuple representing the dimension of object (width, height)
        """
        half_width = 0.5 * (object_dims.value[0])
        half_length = 0.5 * (object_dims.value[1])
        local_verts = np.array(
            [
                [+half_length, +half_width, 0.0],
                [+half_length, -half_width, 0.0],
                [-half_length, -half_width, 0.0],
                [-half_length, +half_width, 0.0],
            ]
        )

        object_position = np.array(
            [object_pose.position.x, object_pose.position.y, object_pose.position.z]
        )
        object_orientation = np.array(
            [
                object_pose.orientation.x,
                object_pose.orientation.y,
                object_pose.orientation.z,
                object_pose.orientation.w,
            ]
        )
        return [object_position + apply_orientation(object_orientation, p) for p in local_verts]

    def add_noncollidable_object(self, object_name):
        """
        Add object_name as non-collidable object

        Args:
            object_name (str): the object name to add to non-collidable object list
        """
        with self.noncollidable_object_lock:
            self.noncollidable_objects.add(object_name)

    def remove_noncollidable_object(self, object_name):
        """
        Remove object_name from non-collidable object name list

        Args:
            object_name (str): the object_name to remove from non-collidable list
        """
        with self.noncollidable_object_lock:
            self.noncollidable_objects.discard(object_name)

    def is_object_collidable(self, object_name):
        """
        Check whether object with given object_name is collidable or not
        Args:
            object_name: name of object to check

        Returns: True if collidable otherwise false
        """
        with self.noncollidable_object_lock:
            return object_name not in self.noncollidable_objects

    def get_collided_object_name(self, racecar_wheel_points, racecar_name):
        """Get object name that racecar collide into

        Args:
            racecar_wheel_points (list): List of points that specifies
                the wheels of the training car
            racecar_name (string): racecar name

        Returns:
            string: Crashed object name if there is a crashed object. Otherwise ''

        Raises:
            GenericRolloutException: Unable to detect collision
        """
        try:
            with self.noncollidable_object_lock:
                noncollidable_objects = self.noncollidable_objects.copy()
            for object_name in self.object_poses.keys():
                if object_name != racecar_name and object_name not in noncollidable_objects:
                    object_pose = self.object_poses[object_name]
                    object_dims = self.object_dims[object_name]
                    object_boundary = Polygon(
                        TrackData.get_object_bounding_rect(object_pose, object_dims)
                    )
                    if any([object_boundary.contains(p) for p in racecar_wheel_points]):
                        return object_name
            return ""
        except Exception as ex:
            raise GenericRolloutException("Unable to detect collision {}".format(ex))

    def get_objects_in_camera_frustums(self, agent_name, car_pose, object_order=None):
        """Returns list of tuple (idx, object.pose) for the objects
        that are in camera frustums"""

        frustum = FrustumManager.get_instance().get(agent_name=agent_name)
        frustum.update(car_pose)
        objects_in_frustum = []

        object_order = (
            object_order if object_order is not None else range(len(self.object_poses.values()))
        )

        object_poses = [pose for pose in self.object_poses.values()]
        object_dims = [pose for pose in self.object_dims.values()]
        for idx, order_idx in enumerate(object_order):
            object_pose = object_poses[order_idx]
            object_dim = object_dims[order_idx]

            object_position = np.array(
                [object_pose.position.x, object_pose.position.y, object_pose.position.z]
            )
            object_points = TrackData.get_object_bounding_rect(object_pose, object_dim)
            object_points.append(object_position)
            # Check collision between frustum and object points
            # object points contains object position + points in bounding rectangle on the floor
            # Camera pitch and roll are fixed so checking bounding rectangle on the floor should be good enough
            # One edge case that this detection can miss is similar as below:
            #     FFFFFFFFFFFFFFF
            #     F             F
            #   AAAFAAAAAAAAAAAFAAAAAAAAAAAAAA
            #   A   F         F              A
            #   A    F       F  X            A
            #   AAAAAAFAAAAAFAAAAAAAAAAAAAAAAA
            #          F   F
            #           FFF
            # F = frustum / A = object / X = center of mass of the object
            if any([frustum.is_visible(p) for p in object_points]):
                objects_in_frustum.append((idx, object_pose))
        return objects_in_frustum

    def get_racecar_start_pose(self, racecar_idx, racer_num, start_position):
        """get initial car pose on the track for spawning the follow car camera

        Args:
            racecar_idx (int): racecar index for getting start pose
            racer_num (int): total number of racecars
            start_position (float): racecar start position wrt starting line

        Returns:
            Pose: car model pose
        """
        # Compute the starting position and heading
        # single racer: spawn at centerline
        if racer_num == 1:
            car_model_pose = self.center_line.interpolate_pose(
                distance=0.0, normalized=True, finite_difference=FiniteDifference.FORWARD_DIFFERENCE
            )
        # multi racers: spawn odd car at inner lane and even car at outer lane
        else:
            lane = self.inner_lane if racecar_idx % 2 else self.outer_lane
            car_model_pose = lane.interpolate_pose(
                distance=start_position,
                normalized=False,
                finite_difference=FiniteDifference.FORWARD_DIFFERENCE,
            )
        return car_model_pose
