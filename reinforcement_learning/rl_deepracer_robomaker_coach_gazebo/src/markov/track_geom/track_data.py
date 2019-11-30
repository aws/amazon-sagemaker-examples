'''This module is used to manage the track related data'''
from collections import OrderedDict
from enum import Enum, unique
import math
import os
import threading
import numpy as np
import rospkg
import rospy

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import GetLinkState, GetModelState
from geometry_msgs.msg import Pose
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import LinearRing, LineString

from markov.agent_ctrl.constants import RewardParam
from markov.cameras.frustum import Frustum
from markov.track_geom.constants import TrackNearPnts, TrackNearDist, AgentPos, \
                                        GET_LINK_STATE, GET_MODEL_STATE
from markov.track_geom.utils import euler_to_quaternion, apply_orientation, find_prev_next, quaternion_to_euler
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.deepracer_exceptions import GenericRolloutException


@unique
class FiniteDifference(Enum):
    CENTRAL_DIFFERENCE = 1
    FORWARD_DIFFERENCE = 2

class TrackLine(object):
    def __init__(self, line):
        self.line = line
        self.ndists = [self.line.project(Point(p), normalized=True)
                       for p in self.line.coords[:-1]] + [1.0]

    def __getattr__(self, name):
        return getattr(self.line, name)

    def find_prev_next_waypoints(self, distance, normalized=False, reverse_dir=False):
        ndist = distance if normalized else distance / self.line.length
        return find_prev_next(self.ndists, ndist, reverse_dir)

    def interpolate_yaw(self, distance, normalized=False, reverse_dir=False, position=None,
                        finite_difference=FiniteDifference.CENTRAL_DIFFERENCE):
        prev_index, next_index = self.find_prev_next_waypoints(distance, normalized, reverse_dir)
        if finite_difference == FiniteDifference.CENTRAL_DIFFERENCE:
            yaw = math.atan2(self.line.coords[next_index][1] - self.line.coords[prev_index][1],
                             self.line.coords[next_index][0] - self.line.coords[prev_index][0])
        elif finite_difference == FiniteDifference.FORWARD_DIFFERENCE:
            if not position: position = self.interpolate(distance, normalized)
            yaw = math.atan2(self.line.coords[next_index][1] - position.y,
                             self.line.coords[next_index][0] - position.x)
        else:
            raise ValueError("Unrecognized FiniteDifference enum value")
        return yaw

    def interpolate_pose(self, distance, normalized=False, reverse_dir=False,
                         finite_difference=FiniteDifference.CENTRAL_DIFFERENCE):
        pose = Pose()
        position = self.interpolate(distance, normalized)
        yaw = self.interpolate_yaw(distance, normalized, reverse_dir, position, finite_difference)
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
    '''This class is responsible for managing all the track geometry, the object should
       be created and shared between agents on the track
    '''
    # The track data will be a singelton to prevent copying across multiple agents
    _instance_ = None

    @staticmethod
    def get_instance():
        '''Method for geting a reference to the track data object'''
        if TrackData._instance_ is None:
            TrackData()
        return TrackData._instance_

    def __init__(self):
        '''Instantiates the class and creates clients for the relevant ROS services'''
        if TrackData._instance_ is not None:
            raise GenericRolloutException("Attempting to construct multiple tack data objects")

        rospy.wait_for_service(GET_LINK_STATE)
        rospy.wait_for_service(GET_MODEL_STATE)

        self._get_link_state_ = ServiceProxyWrapper(GET_LINK_STATE, GetLinkState)
        self._get_model_state_ = ServiceProxyWrapper(GET_MODEL_STATE, GetModelState)
        try:
            rospack = rospkg.RosPack()
            deepracer_path = rospack.get_path("deepracer_simulation_environment")
            waypoints_path = os.path.join(deepracer_path, "routes",
                                          "{}.npy".format(rospy.get_param("WORLD_NAME")))
            self._is_bot_car_ = int(rospy.get_param("NUMBER_OF_BOT_CARS", 0)) > 0
            self._bot_car_speed_ = float(rospy.get_param("BOT_CAR_SPEED", 0.0))
            waypoints = np.load(waypoints_path)

            self.is_loop = np.all(waypoints[0,:] == waypoints[-1,:])
            poly_func = LinearRing if self.is_loop else LineString
            self._center_line_ = TrackLine(poly_func(waypoints[:, 0:2]))
            self._inner_border_ = TrackLine(poly_func(waypoints[:, 2:4]))
            self._outer_border_ = TrackLine(poly_func(waypoints[:, 4:6]))
            self._inner_lane_ = TrackLine(poly_func((waypoints[:,2:4] + waypoints[:,0:2])/2))
            self._outer_lane_ = TrackLine(poly_func((waypoints[:,4:6] + waypoints[:,0:2])/2))
            if self.is_loop:
                self._left_poly_ = Polygon(self._center_line_, [self._inner_border_])
                self._road_poly_ = Polygon(self._outer_border_, [self._inner_border_])
            else:
                self._left_poly_ = Polygon(np.vstack((self._center_line_.line,
                                                      np.flipud(self._inner_border_))))
                self._road_poly_ = Polygon(np.vstack((self._outer_border_,
                                                      np.flipud(self._inner_border_))))

            self.car_ndist = 0.0 # TEMPORARY -- REMOVE THIS
            self.object_poses = OrderedDict()
            self.object_dims = OrderedDict()
            rospy.Subscriber('/gazebo/model_states', ModelStates, self._update_objects)

            # There should only be one track data object
            TrackData._instance_ = self

            # declare a lock to prevent read and write at the same time
            self._lock_ = threading.Lock()

        except Exception as ex:
            raise GenericRolloutException('Failed to create track data: {}'.format(ex))

    def _update_objects(self, model_states):
        with self._lock_:
            for name, pose in zip(model_states.name, model_states.pose):
                if name in self.object_poses:
                    self.object_poses[name] = pose

    def initialize_object(self, name, initial_pose, object_dimensions):
        self.object_poses[name] = initial_pose
        self.object_dims[name] = object_dimensions

    def reset_object(self, name, initial_pose):
        if name in self.object_poses:
            self.object_poses[name] = initial_pose
        else:
            raise GenericRolloutException('Failed to reset unrecognized object: {}'.format(name))

    def find_prev_next_waypoints(self, distance, normalized=False, reverse_dir=False):
        return self._center_line_.find_prev_next_waypoints(distance, normalized, reverse_dir)

    @staticmethod
    def get_nearest_dist(near_pnt_dict, model_point):
        '''Auxiliary method for computing the distance to the nearest points given in
           near_pnt_dict.
           near_pnt_dict - Dictionary containing the keys specified in TrackNearPnts
           model_point - Object returned by calling the model state service which contains
                         the position data for the agent
        '''
        try:
            dist_from_cent = near_pnt_dict[TrackNearPnts.NEAR_PNT_CENT.value].distance(model_point)
            dist_from_in = near_pnt_dict[TrackNearPnts.NEAR_PNT_IN.value].distance(model_point)
            dist_from_out = near_pnt_dict[TrackNearPnts.NEAR_PNT_OUT.value].distance(model_point)

            return {TrackNearDist.NEAR_DIST_CENT.value : dist_from_cent,
                    TrackNearDist.NEAR_DIST_IN.value : dist_from_in,
                    TrackNearDist.NEAR_DIST_OUT.value : dist_from_out}
        except Exception as ex:
            raise GenericRolloutException("Unable to compute nearest distance: {}".format(ex))

    def get_agent_pos(self, agent_name, link_name_list, relative_pos):
        '''Returns a dictionary with the keys defined in AgentPos which contains
           the position of the agent on the track, the location of the desired
           links, and the orientation of the agent.
           agent_name - String with the name of the agent
           link_name_list - List of strings containing the name of the links whose
                            positions are to be retrieved.
            relative_pos - List containing the x-y relative position of the front of
                           the agent
        '''
        try:
            #Request the model state from gazebo
            model_state = self._get_model_state_(agent_name, '')
            #Compute the model's orientation
            model_orientation = np.array([model_state.pose.orientation.x,
                                          model_state.pose.orientation.y,
                                          model_state.pose.orientation.z,
                                          model_state.pose.orientation.w])
            #Compute the model's location relative to the front of the agent
            model_location = np.array([model_state.pose.position.x,
                                       model_state.pose.position.y,
                                       model_state.pose.position.z]) + \
                             apply_orientation(model_orientation, np.array(relative_pos))
            model_point = Point(model_location[0], model_location[1])
            #Grab the location of the links
            make_link_points = lambda link: Point(link.link_state.pose.position.x,
                                                  link.link_state.pose.position.y)
            link_points = [make_link_points(self._get_link_state_(name, ''))
                           for name in link_name_list]

            return {AgentPos.ORIENTATION.value : model_orientation,
                    AgentPos.POINT.value : model_point,
                    AgentPos.LINK_POINTS.value :link_points}
        except Exception as ex:
            raise GenericRolloutException("Unable to get position: {}".format(ex))

    def get_track_length(self):
        '''Returns the length of the track'''
        try:
            return self._center_line_.length
        except  Exception as ex:
            raise GenericRolloutException("Unable to get track lenght: {}".format(ex))

    def get_way_pnts(self):
        '''Returns a list containing all the way points'''
        try:
            return list(self._center_line_.coords)
        except  Exception as ex:
            raise GenericRolloutException("Unable to get way points: {}".format(ex))

    def get_norm_dist(self, model_point):
        '''Returns the normalized position of the agent relative to the track
           model_point - Object returned by calling the model state service which contains
                         the position data for the agent
        '''
        try:
            return self._center_line_.project(model_point, normalized=True)
        except Exception as ex:
            raise GenericRolloutException("Unable to get norm dist: {}".format(ex))

    def get_nearest_points(self, model_point):
        '''Returns a dictionary with the keys specified in TrackNearPnts containing
           the nearest way points to the agent.
           model_point - Object returned by calling the model state service which contains
                         the position data for the agent
        '''
        try:
            near_pnt_ctr = \
                self._center_line_.interpolate(self.get_norm_dist(model_point), normalized=True)
            near_pnt_in = \
                self._inner_border_.interpolate(self._inner_border_.project(near_pnt_ctr))
            near_pnt_out = \
                self._outer_border_.interpolate(self._outer_border_.project(near_pnt_ctr))

            return {TrackNearPnts.NEAR_PNT_CENT.value : near_pnt_ctr,
                    TrackNearPnts.NEAR_PNT_IN.value : near_pnt_in,
                    TrackNearPnts.NEAR_PNT_OUT.value : near_pnt_out}
        except Exception as ex:
            raise GenericRolloutException("Unable to get nearest points: {}".format(ex))

    def get_object_reward_params(self, model_point, model_heading, current_progress, reverse_dir):
        '''Returns a dictionary with object-related reward function params.'''
        with self._lock_:
            try:
                object_locations = [[pose.position.x, pose.position.y]
                                    for pose in self.object_poses.values()]
                object_poses = [pose for pose in self.object_poses.values()]
                if not object_locations:
                    return {}

                # Sort the object locations based on projected distance
                object_pdists = [self._center_line_.project(Point(p)) for p in object_locations]
                object_order = np.argsort(object_pdists)
                object_pdists = [object_pdists[i] for i in object_order]
                object_locations = [object_locations[i] for i in object_order]
                object_headings = []
                object_speeds = []
                object_headings = [0.0 for _ in object_order] 
                object_speeds = [0.0 for _ in object_order]
                if self._is_bot_car_:
                    for i in object_order:
                        object_pose = object_poses[i]
                        _, _, yaw = quaternion_to_euler(x=object_pose.orientation.x,
                                                        y=object_pose.orientation.y,
                                                        z=object_pose.orientation.z,
                                                        w=object_pose.orientation.w)
                        object_headings[i] = yaw
                        object_speeds[i] = self._bot_car_speed_

                # Find the prev/next objects
                model_pdist = self._center_line_.project(model_point)
                prev_object_index, next_object_index = find_prev_next(object_pdists, model_pdist,
                                                                      reverse_dir)

                # Figure out which one is the closest
                object_points = [Point([object_location[0], object_location[1]])
                                   for object_location in object_locations]
                prev_object_point = object_points[prev_object_index]
                next_object_point = object_points[next_object_index]
                prev_object_dist = model_point.distance(prev_object_point)
                next_object_dist = model_point.distance(next_object_point)
                if prev_object_dist < next_object_dist:
                    closest_object_point = prev_object_point
                else:
                    closest_object_point = next_object_point

                # Figure out whether objects is left of center based on direction
                objects_left_of_center = [self._left_poly_.contains(p) ^ reverse_dir \
                                          for p in object_points]

                # Figure out which lane the model is in
                model_nearest_pnts_dict = self.get_nearest_points(model_point)
                model_nearest_dist_dict = self.get_nearest_dist(model_nearest_pnts_dict, model_point)
                model_is_inner = \
                    model_nearest_dist_dict[TrackNearDist.NEAR_DIST_IN.value] < \
                    model_nearest_dist_dict[TrackNearDist.NEAR_DIST_OUT.value]

                # Figure out which lane the object is in
                closest_object_nearest_pnts_dict = self.get_nearest_points(closest_object_point)
                closest_object_nearest_dist_dict = \
                    self.get_nearest_dist(closest_object_nearest_pnts_dict, closest_object_point)
                closest_object_is_inner = \
                    closest_object_nearest_dist_dict[TrackNearDist.NEAR_DIST_IN.value] < \
                    closest_object_nearest_dist_dict[TrackNearDist.NEAR_DIST_OUT.value]

                objects_in_camera = self.get_objects_in_camera_frustums(object_order)
                is_next_object_in_camera = any(object_in_camera_idx == next_object_index
                                               for object_in_camera_idx, _ in objects_in_camera)
                # Determine if they are in the same lane
                return {RewardParam.CLOSEST_OBJECTS.value[0]: [prev_object_index, next_object_index],
                        RewardParam.OBJECT_LOCATIONS.value[0]: object_locations,
                        RewardParam.OBJECTS_LEFT_OF_CENTER.value[0]: objects_left_of_center,
                        RewardParam.OBJECT_SPEEDS.value[0]: object_speeds,
                        RewardParam.OBJECT_HEADINGS.value[0]: object_headings,
                        RewardParam.OBJECT_CENTERLINE_PROJECTION_DISTANCES.value[0]: object_pdists,
                        RewardParam.OBJECT_IN_CAMERA.value[0]: is_next_object_in_camera
                        }
            except Exception as ex:
                raise GenericRolloutException("Unable to get object reward params: {}".format(ex))

    def get_distance_from_next_and_prev(self, model_point, prev_index, next_index):
        '''Returns a tuple, where the first value is the distance to the given previous points
           and the second value is the distance to given next point.
           model_point - Object returned by calling the model state service which contains
                         the position data for the agent
           prev_index - Integer representing the index of the previous point
           next_index - Integer representing the index of the next point
        '''
        try:
            dist_from_prev = model_point.distance(Point(self._center_line_.coords[prev_index]))
            dist_from_next = model_point.distance(Point(self._center_line_.coords[next_index]))
            return dist_from_prev, dist_from_next
        except Exception as ex:
            raise GenericRolloutException("Unable to get distance to prev and next points: {}".format(ex))

    def points_on_track(self, points):
        '''Returns a boolean list, where entries of true represent a point in the points list
           being on the track, and values of false represent a point in the points list being
           of the track.
           points - List of points who will be checked for being on or off the track
        '''
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
        local_verts = np.array([[+half_length, +half_width, 0.0],
                                [+half_length, -half_width, 0.0],
                                [-half_length, -half_width, 0.0],
                                [-half_length, +half_width, 0.0]])

        object_position = np.array([object_pose.position.x,
                                    object_pose.position.y,
                                    object_pose.position.z])
        object_orientation = np.array([object_pose.orientation.x,
                                       object_pose.orientation.y,
                                       object_pose.orientation.z,
                                       object_pose.orientation.w])
        return [object_position
                + apply_orientation(object_orientation, p)
                for p in local_verts]

    def is_racecar_collided(self, racecar_wheel_points):
        '''Returns a true if there is a collision between the racecar and an object
           racecar_wheel_points - List of points that specifies the wheels of the training car
        '''
        try:
            for object_pose, object_dims in zip(self.object_poses.values(), self.object_dims.values()):
                object_boundary = Polygon(TrackData.get_object_bounding_rect(object_pose, object_dims))
                if any([object_boundary.contains(p) for p in racecar_wheel_points]):
                    return True
        except Exception as ex:
            raise GenericRolloutException("Unable to detect collision {}".format(ex))

    def get_objects_in_camera_frustums(self, object_order=None):
        """Returns list of tuple (idx, object.pose) for the objects
        that are in camera frustums"""

        frustum = Frustum.get_instance()
        frustum.update()
        objects_in_frustum = []

        object_order = object_order if object_order is not None else range(len(self.object_poses.values()))

        object_poses = [pose for pose in self.object_poses.values()]
        object_dims = [pose for pose in self.object_dims.values()]
        for idx, order_idx in enumerate(object_order):
            object_pose = object_poses[order_idx]
            object_dim = object_dims[order_idx]

            object_position = np.array([object_pose.position.x,
                                        object_pose.position.y,
                                        object_pose.position.z])
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
