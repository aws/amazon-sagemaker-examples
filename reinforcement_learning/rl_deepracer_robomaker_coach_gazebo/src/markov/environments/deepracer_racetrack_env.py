from __future__ import print_function

import bisect
import boto3
import json
import logging
import math
import os
import time
import traceback
import sys
from collections import OrderedDict

import gym
import queue
import numpy as np
from gym import spaces
from PIL import Image
from markov import utils

logger = utils.Logger(__name__, logging.INFO).get_logger()

# Type of worker
SIMULATION_WORKER = "SIMULATION_WORKER"
SAGEMAKER_TRAINING_WORKER = "SAGEMAKER_TRAINING_WORKER"

node_type = os.environ.get("NODE_TYPE", SIMULATION_WORKER)
if node_type == SIMULATION_WORKER:
    import rospy
    from std_msgs.msg import Float64
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import GetLinkState, GetModelState, JointRequest
    from scipy.spatial.transform import Rotation
    from sensor_msgs.msg import Image as sensor_image
    from shapely.geometry import Point, Polygon
    from shapely.geometry.polygon import LinearRing, LineString
    from deepracer_simulation_environment.srv import GetWaypointSrv, ResetCarSrv

# Type of job
TRAINING_JOB = 'TRAINING'
EVALUATION_JOB = 'EVALUATION'

# Dimensions of the input training image
TRAINING_IMAGE_SIZE = (160, 120)

# Local offset of the front of the car
RELATIVE_POSITION_OF_FRONT_OF_CAR = [0.14, 0, 0]

# Normalized track distance to move with each reset
ROUND_ROBIN_ADVANCE_DIST = 0.05

# Reward to give the car when it "crashes"
CRASHED = 1e-8

# Size of the image queue buffer, we want this to be one so that we consume 1 image
# at a time, but may want to change this as we add more algorithms
IMG_QUEUE_BUF_SIZE = 1

# List of required velocity topics, one topic per wheel
VELOCITY_TOPICS = ['/racecar/left_rear_wheel_velocity_controller/command',
                   '/racecar/right_rear_wheel_velocity_controller/command',
                   '/racecar/left_front_wheel_velocity_controller/command',
                   '/racecar/right_front_wheel_velocity_controller/command']

# List of required steering hinges
STEERING_TOPICS = ['/racecar/left_steering_hinge_position_controller/command',
                   '/racecar/right_steering_hinge_position_controller/command']

# List of all effort joints
EFFORT_JOINTS = ['/racecar/left_rear_wheel_joint', '/racecar/right_rear_wheel_joint',
                 '/racecar/left_front_wheel_joint','/racecar/right_front_wheel_joint',
                 '/racecar/left_steering_hinge_joint','/racecar/right_steering_hinge_joint']
# Radius of the wheels of the car in meters
WHEEL_RADIUS = 0.1

# The number of steps to wait before checking if the car is stuck
# This number should corespond to the camera FPS, since it is pacing the
# step rate.
NUM_STEPS_TO_CHECK_STUCK = 15

### Gym Env ###
class DeepRacerRacetrackEnv(gym.Env):

    def __init__(self):

        # Create the observation space
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(TRAINING_IMAGE_SIZE[1], TRAINING_IMAGE_SIZE[0], 3),
                                            dtype=np.uint8)
        # Create the action space
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([+1, +1]), dtype=np.float32)

        if node_type == SIMULATION_WORKER:

            # ROS initialization
            rospy.init_node('rl_coach', anonymous=True)

            # wait for required services
            rospy.wait_for_service('/deepracer_simulation_environment/get_waypoints')
            rospy.wait_for_service('/deepracer_simulation_environment/reset_car')
            rospy.wait_for_service('/gazebo/get_model_state')
            rospy.wait_for_service('/gazebo/get_link_state')
            rospy.wait_for_service('/gazebo/clear_joint_forces')

            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
            self.clear_forces_client = rospy.ServiceProxy('/gazebo/clear_joint_forces',
                                                          JointRequest)
            self.reset_car_client = rospy.ServiceProxy('/deepracer_simulation_environment/reset_car',
                                                       ResetCarSrv)
            get_waypoints_client = rospy.ServiceProxy('/deepracer_simulation_environment/get_waypoints',
                                                      GetWaypointSrv)

            # Create the publishers for sending speed and steering info to the car
            self.velocity_pub_dict = OrderedDict()
            self.steering_pub_dict = OrderedDict()

            for topic in VELOCITY_TOPICS:
                self.velocity_pub_dict[topic] = rospy.Publisher(topic, Float64, queue_size=1)

            for topic in STEERING_TOPICS:
                self.steering_pub_dict[topic] = rospy.Publisher(topic, Float64, queue_size=1)

            # Read in parameters
            self.world_name = rospy.get_param('WORLD_NAME')
            self.job_type = rospy.get_param('JOB_TYPE')
            self.aws_region = rospy.get_param('AWS_REGION')
            self.metrics_s3_bucket = rospy.get_param('METRICS_S3_BUCKET')
            self.metrics_s3_object_key = rospy.get_param('METRICS_S3_OBJECT_KEY')
            self.metrics = []
            self.simulation_job_arn = 'arn:aws:robomaker:' + self.aws_region + ':' + \
                                      rospy.get_param('ROBOMAKER_SIMULATION_JOB_ACCOUNT_ID') + \
                                      ':simulation-job/' + rospy.get_param('AWS_ROBOMAKER_SIMULATION_JOB_ID')

            if self.job_type == TRAINING_JOB:
                from custom_files.customer_reward_function import reward_function
                self.reward_function = reward_function
                self.metric_name = rospy.get_param('METRIC_NAME')
                self.metric_namespace = rospy.get_param('METRIC_NAMESPACE')
                self.training_job_arn = rospy.get_param('TRAINING_JOB_ARN')
                self.target_number_of_episodes = rospy.get_param('NUMBER_OF_EPISODES')
                self.target_reward_score = rospy.get_param('TARGET_REWARD_SCORE')
            else:
                from markov.defaults import reward_function
                self.reward_function = reward_function
                self.number_of_trials = 0
                self.target_number_of_trials = rospy.get_param('NUMBER_OF_TRIALS')

            # Request the waypoints
            waypoints = None
            try:
                resp = get_waypoints_client()
                waypoints = np.array(resp.waypoints).reshape(resp.row, resp.col)
            except Exception as ex:
                utils.json_format_logger("Unable to retrieve waypoints: {}".format(ex),
                             **utils.build_system_error_dict(utils.SIMAPP_ENVIRONMENT_EXCEPTION,
                                                             utils.SIMAPP_EVENT_ERROR_CODE_500))

            is_loop = np.all(waypoints[0,:] == waypoints[-1,:])
            if is_loop:
                self.center_line = LinearRing(waypoints[:,0:2])
                self.inner_border = LinearRing(waypoints[:,2:4])
                self.outer_border = LinearRing(waypoints[:,4:6])
                self.road_poly = Polygon(self.outer_border, [self.inner_border])
            else:
                self.center_line = LineString(waypoints[:,0:2])
                self.inner_border = LineString(waypoints[:,2:4])
                self.outer_border = LineString(waypoints[:,4:6])
                self.road_poly = Polygon(np.vstack((self.outer_border, np.flipud(self.inner_border))))
            self.center_dists = [self.center_line.project(Point(p), normalized=True) for p in self.center_line.coords[:-1]] + [1.0]
            self.track_length = self.center_line.length
            # Queue used to maintain image consumption synchronicity
            self.image_queue = queue.Queue(IMG_QUEUE_BUF_SIZE)
            rospy.Subscriber('/camera/zed/rgb/image_rect_color', sensor_image, self.callback_image)

            # Initialize state data
            self.episodes = 0
            self.start_ndist = 0.0
            self.reverse_dir = False
            self.change_start = rospy.get_param('CHANGE_START_POSITION', (self.job_type == TRAINING_JOB))
            self.alternate_dir = rospy.get_param('ALTERNATE_DRIVING_DIRECTION', False)
            self.is_simulation_done = False
            self.steering_angle = 0
            self.speed = 0
            self.action_taken = 0
            self.prev_progress = 0
            self.prev_point = Point(0, 0)
            self.prev_point_2 = Point(0, 0)
            self.next_state = None
            self.reward = None
            self.reward_in_episode = 0
            self.done = False
            self.steps = 0
            self.simulation_start_time = 0
            self.allow_servo_step_signals = False

    def reset(self):
        if node_type == SAGEMAKER_TRAINING_WORKER:
            return self.observation_space.sample()
        
        # Simulation is done - so RoboMaker will start to shut down the app.
        # Till RoboMaker shuts down the app, do nothing more else metrics may show unexpected data.
        if (node_type == SIMULATION_WORKER) and self.is_simulation_done:
            while True:
                time.sleep(1)

        self.steering_angle = 0
        self.speed = 0
        self.action_taken = 0
        self.prev_progress = 0
        self.prev_point = Point(0, 0)
        self.prev_point_2 = Point(0, 0)
        self.next_state = None
        self.reward = None
        self.reward_in_episode = 0
        self.done = False
        # Reset the car and record the simulation start time
        if self.allow_servo_step_signals:
            self.send_action(0, 0)

        self.racecar_reset()
        self.steps = 0
        self.simulation_start_time = time.time()
        self.infer_reward_state(0, 0)

        return self.next_state

    def set_next_state(self):
        # Make sure the first image is the starting image
        image_data = self.image_queue.get(block=True, timeout=None)
        # Read the image and resize to get the state
        image = Image.frombytes('RGB', (image_data.width, image_data.height), image_data.data, 'raw', 'RGB', 0, 1)
        image = image.resize(TRAINING_IMAGE_SIZE, resample=2)
        self.next_state = np.array(image)

    def racecar_reset(self):
        try:
            for joint in EFFORT_JOINTS:
                self.clear_forces_client(joint)
            prev_index, next_index = self.find_prev_next_waypoints(self.start_ndist)
            self.reset_car_client(self.start_ndist, next_index)
            # First clear the queue so that we set the state to the start image
            _ = self.image_queue.get(block=True, timeout=None)
            self.set_next_state()

        except Exception as ex:
            utils.json_format_logger("Unable to reset the car: {}".format(ex),
                         **utils.build_system_error_dict(utils.SIMAPP_ENVIRONMENT_EXCEPTION,
                                                         utils.SIMAPP_EVENT_ERROR_CODE_500))

    def set_allow_servo_step_signals(self, allow_servo_step_signals):
        self.allow_servo_step_signals = allow_servo_step_signals

    def step(self, action):
        if node_type == SAGEMAKER_TRAINING_WORKER:
            return self.observation_space.sample(), 0, False, {}

        # Initialize next state, reward, done flag
        self.next_state = None
        self.reward = None
        self.done = False

        # Send this action to Gazebo and increment the step count
        self.steering_angle = float(action[0])
        self.speed = float(action[1])
        if self.allow_servo_step_signals:
            self.send_action(self.steering_angle, self.speed)
        self.steps += 1

        # Compute the next state and reward
        self.infer_reward_state(self.steering_angle, self.speed)
        return self.next_state, self.reward, self.done, {}

    def callback_image(self, data):
        try:
            self.image_queue.put_nowait(data)
        except queue.Full:
            pass
        except Exception as ex:
            utils.json_format_logger("Error retrieving frame from gazebo: {}".format(ex),
                       **utils.build_system_error_dict(utils.SIMAPP_ENVIRONMENT_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_500))

    def send_action(self, steering_angle, speed):
        # Simple v/r to computes the desired rpm
        wheel_rpm = speed/WHEEL_RADIUS

        for _, pub in self.velocity_pub_dict.items():
            pub.publish(wheel_rpm)

        for _, pub in self.steering_pub_dict.items():
            pub.publish(steering_angle)

    def infer_reward_state(self, steering_angle, speed):
        try:
            self.set_next_state()
        except Exception as ex:
            utils.json_format_logger("Unable to retrieve image from queue: {}".format(ex),
                       **utils.build_system_error_dict(utils.SIMAPP_ENVIRONMENT_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_500))

        # Read model state from Gazebo
        model_state = self.get_model_state('racecar', '')
        model_orientation = Rotation.from_quat([
            model_state.pose.orientation.x,
            model_state.pose.orientation.y,
            model_state.pose.orientation.z,
            model_state.pose.orientation.w])
        model_location = np.array([
            model_state.pose.position.x,
            model_state.pose.position.y,
            model_state.pose.position.z]) + \
            model_orientation.apply(RELATIVE_POSITION_OF_FRONT_OF_CAR)
        model_point = Point(model_location[0], model_location[1])
        model_heading = model_orientation.as_euler('zyx')[0]

        # Read the wheel locations from Gazebo
        left_rear_wheel_state = self.get_link_state('racecar::left_rear_wheel', '')
        left_front_wheel_state = self.get_link_state('racecar::left_front_wheel', '')
        right_rear_wheel_state = self.get_link_state('racecar::right_rear_wheel', '')
        right_front_wheel_state = self.get_link_state('racecar::right_front_wheel', '')
        wheel_points = [
            Point(left_rear_wheel_state.link_state.pose.position.x,
                  left_rear_wheel_state.link_state.pose.position.y),
            Point(left_front_wheel_state.link_state.pose.position.x,
                  left_front_wheel_state.link_state.pose.position.y),
            Point(right_rear_wheel_state.link_state.pose.position.x,
                  right_rear_wheel_state.link_state.pose.position.y),
            Point(right_front_wheel_state.link_state.pose.position.x,
                  right_front_wheel_state.link_state.pose.position.y)
        ]

        # Project the current location onto the center line and find nearest points
        current_ndist = self.center_line.project(model_point, normalized=True)
        prev_index, next_index = self.find_prev_next_waypoints(current_ndist)
        distance_from_prev = model_point.distance(Point(self.center_line.coords[prev_index]))
        distance_from_next = model_point.distance(Point(self.center_line.coords[next_index]))
        closest_waypoint_index = (prev_index, next_index)[distance_from_next < distance_from_prev]

        # Compute distance from center and road width
        nearest_point_center = self.center_line.interpolate(current_ndist, normalized=True)
        nearest_point_inner = self.inner_border.interpolate(self.inner_border.project(nearest_point_center))
        nearest_point_outer = self.outer_border.interpolate(self.outer_border.project(nearest_point_center))
        distance_from_center = nearest_point_center.distance(model_point)
        distance_from_inner = nearest_point_inner.distance(model_point)
        distance_from_outer = nearest_point_outer.distance(model_point)
        track_width = nearest_point_inner.distance(nearest_point_outer)
        is_left_of_center = (distance_from_outer < distance_from_inner) if self.reverse_dir \
            else (distance_from_inner < distance_from_outer)

        # Convert current progress to be [0,100] starting at the initial waypoint
        if self.reverse_dir:
            current_progress = self.start_ndist - current_ndist
        else:
            current_progress = current_ndist - self.start_ndist
        if current_progress < 0.0: current_progress = current_progress + 1.0
        current_progress = 100 * current_progress
        if current_progress < self.prev_progress:
            # Either: (1) we wrapped around and have finished the track,
            delta1 = current_progress + 100 - self.prev_progress
            # or (2) for some reason the car went backwards (this should be rare)
            delta2 = self.prev_progress - current_progress
            current_progress = (self.prev_progress, 100)[delta1 < delta2]

        # Car is off track if all wheels are outside the borders
        wheel_on_track = [self.road_poly.contains(p) for p in wheel_points]
        all_wheels_on_track = all(wheel_on_track)
        any_wheels_on_track = any(wheel_on_track)

        # Compute the reward
        if any_wheels_on_track:
            done = False
            params = {
                'all_wheels_on_track': all_wheels_on_track,
                'x': model_point.x,
                'y': model_point.y,
                'heading': model_heading * 180.0 / math.pi,
                'distance_from_center': distance_from_center,
                'progress': current_progress,
                'steps': self.steps,
                'speed': speed,
                'steering_angle': steering_angle * 180.0 / math.pi,
                'track_width': track_width,
                'waypoints': list(self.center_line.coords),
                'closest_waypoints': [prev_index, next_index],
                'is_left_of_center': is_left_of_center,
                'is_reversed': self.reverse_dir
            }
            try:
                reward = float(self.reward_function(params))
            except Exception as e:
                utils.json_format_logger("Exception {} in customer reward function. Job failed!".format(e),
                          **utils.build_user_error_dict(utils.SIMAPP_SIMULATION_WORKER_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_400))
                traceback.print_exc()
                sys.exit(1)
        else:
            done = True
            reward = CRASHED

        # Reset if the car position hasn't changed in the last 2 steps
        prev_pnt_dist = min(model_point.distance(self.prev_point), model_point.distance(self.prev_point_2))

        if prev_pnt_dist <= 0.0001 and self.steps % NUM_STEPS_TO_CHECK_STUCK == 0:
            done = True
            reward = CRASHED  # stuck

        # Simulation jobs are done when progress reaches 100
        if current_progress >= 100:
            done = True

        # Keep data from the previous step around
        self.prev_point_2 = self.prev_point
        self.prev_point = model_point
        self.prev_progress = current_progress

        # Set the reward and done flag
        self.reward = reward
        self.reward_in_episode += reward
        self.done = done

        # Trace logs to help us debug and visualize the training runs
        # btown TODO: This should be written to S3, not to CWL.
        logger.info('SIM_TRACE_LOG:%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%d,%.4f,%s,%s,%.4f,%d,%.2f,%s\n' % (
            self.episodes, self.steps, model_location[0], model_location[1], model_heading,
            self.steering_angle,
            self.speed,
            self.action_taken,
            self.reward,
            self.done,
            all_wheels_on_track,
            current_progress,
            closest_waypoint_index,
            self.track_length,
            time.time()))

        # Terminate this episode when ready
        if done and node_type == SIMULATION_WORKER:
            self.finish_episode(current_progress)

    def find_prev_next_waypoints(self, ndist):
        if self.reverse_dir:
            next_index = bisect.bisect_left(self.center_dists, ndist) - 1
            prev_index = next_index + 1
            if next_index == -1: next_index = len(self.center_dists) - 1
        else:
            next_index = bisect.bisect_right(self.center_dists, ndist)
            prev_index = next_index - 1
            if next_index == len(self.center_dists): next_index = 0
        return prev_index, next_index

    def stop_car(self):
        self.steering_angle = 0
        self.speed = 0
        self.action_taken = 0
        self.send_action(0, 0)
        self.racecar_reset()

    def finish_episode(self, progress):
        # Increment episode count, update start position and direction
        self.episodes += 1
        if self.change_start:
            self.start_ndist = (self.start_ndist + ROUND_ROBIN_ADVANCE_DIST) % 1.0
        if self.alternate_dir:
            self.reverse_dir = not self.reverse_dir
        # Reset the car
        self.stop_car()

        # Update metrics based on job type
        if self.job_type == TRAINING_JOB:
            self.send_reward_to_cloudwatch(self.reward_in_episode)
            self.update_training_metrics()
            self.write_metrics_to_s3()
            if self.is_training_done():
                self.cancel_simulation_job()
        elif self.job_type == EVALUATION_JOB:
            self.number_of_trials += 1
            self.update_eval_metrics(progress)
            self.write_metrics_to_s3()

    def update_eval_metrics(self, progress):
        eval_metric = {}
        eval_metric['completion_percentage'] = int(progress)
        eval_metric['metric_time'] = int(round(time.time() * 1000))
        eval_metric['start_time'] = int(round(self.simulation_start_time * 1000))
        eval_metric['elapsed_time_in_milliseconds'] = int(round((time.time() - self.simulation_start_time) * 1000))
        eval_metric['trial'] = int(self.number_of_trials)
        self.metrics.append(eval_metric)

    def update_training_metrics(self):
        training_metric = {}
        training_metric['reward_score'] = int(round(self.reward_in_episode))
        training_metric['metric_time'] = int(round(time.time() * 1000))
        training_metric['start_time'] = int(round(self.simulation_start_time * 1000))
        training_metric['elapsed_time_in_milliseconds'] = int(round((time.time() - self.simulation_start_time) * 1000))
        training_metric['episode'] = int(self.episodes)
        self.metrics.append(training_metric)

    def write_metrics_to_s3(self):
        session = boto3.session.Session()
        s3_client = session.client('s3', region_name=self.aws_region)
        metrics_body = json.dumps({'metrics': self.metrics})
        s3_client.put_object(
            Bucket=self.metrics_s3_bucket,
            Key=self.metrics_s3_object_key,
            Body=bytes(metrics_body, encoding='utf-8')
        )

    def is_training_done(self):
        if ((self.target_number_of_episodes > 0) and (self.target_number_of_episodes == self.episodes)) or \
           ((isinstance(self.target_reward_score, (int, float))) and (self.target_reward_score <= self.reward_in_episode)):
            self.is_simulation_done = True
        return self.is_simulation_done

    def cancel_simulation_job(self):
        session = boto3.session.Session()
        robomaker_client = session.client('robomaker', region_name=self.aws_region)
        robomaker_client.cancel_simulation_job(
            job=self.simulation_job_arn
        )

    def send_reward_to_cloudwatch(self, reward):
        session = boto3.session.Session()
        cloudwatch_client = session.client('cloudwatch', region_name=self.aws_region)
        cloudwatch_client.put_metric_data(
            MetricData=[
                {
                    'MetricName': self.metric_name,
                    'Dimensions': [
                        {
                            'Name': 'TRAINING_JOB_ARN',
                            'Value': self.training_job_arn
                        },
                    ],
                    'Unit': 'None',
                    'Value': reward
                },
            ],
            Namespace=self.metric_namespace
        )

class DeepRacerRacetrackCustomActionSpaceEnv(DeepRacerRacetrackEnv):
    def __init__(self):
        DeepRacerRacetrackEnv.__init__(self)
        try:
            # Try loading the custom model metadata (may or may not be present)
            with open('./custom_files/model_metadata.json', 'r') as f:
                model_metadata = json.load(f)
                self.json_actions = model_metadata['action_space']
            logger.info("Loaded action space from file: {}".format(self.json_actions))
        except Exception as ex:
            # Failed to load, fall back on the default action space
            from markov.defaults import model_metadata
            self.json_actions = model_metadata['action_space']
            logger.info("Exception {} on loading custom action space, using default: {}".format(ex, self.json_actions))
        self.action_space = spaces.Discrete(len(self.json_actions))

    def step(self, action):
        self.steering_angle = float(self.json_actions[action]['steering_angle']) * math.pi / 180.0
        self.speed = float(self.json_actions[action]['speed'])
        self.action_taken = action
        return super().step([self.steering_angle, self.speed])
