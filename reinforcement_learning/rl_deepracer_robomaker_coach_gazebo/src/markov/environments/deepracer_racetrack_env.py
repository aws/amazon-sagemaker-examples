from __future__ import print_function

import bisect
import boto3
import json
import logging
import math
import os
import time

import gym
import numpy as np
from gym import spaces
from PIL import Image

logger = logging.getLogger(__name__)

# Type of worker
SIMULATION_WORKER = "SIMULATION_WORKER"
SAGEMAKER_TRAINING_WORKER = "SAGEMAKER_TRAINING_WORKER"

node_type = os.environ.get("NODE_TYPE", SIMULATION_WORKER)
if node_type == SIMULATION_WORKER:
    #subprocess.call("pip install rospy", shell=True)
    
    import rospy
    from ackermann_msgs.msg import AckermannDriveStamped
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import GetLinkState, GetModelState, SetModelState
    from scipy.spatial.transform import Rotation
    from sensor_msgs.msg import Image as sensor_image
    from shapely.geometry import Point, Polygon
    from shapely.geometry.polygon import LinearRing, LineString

# Type of job
TRAINING_JOB = 'TRAINING'
EVALUATION_JOB = 'EVALUATION'

# Sleep intervals
SLEEP_AFTER_RESET_TIME_IN_SECOND = 0.5
SLEEP_BETWEEN_ACTION_AND_REWARD_CALCULATION_TIME_IN_SECOND = 0.1
SLEEP_WAITING_FOR_IMAGE_TIME_IN_SECOND = 0.01

# Dimensions of the input training image
TRAINING_IMAGE_SIZE = (160, 120)

# Local offset of the front of the car
RELATIVE_POSITION_OF_FRONT_OF_CAR = [0.14, 0, 0]

# Normalized track distance to move with each reset
ROUND_ROBIN_ADVANCE_DIST = 0.05

# Reward to give the car when it "crashes"
CRASHED = 1e-8

### Gym Env ###
class DeepRacerRacetrackEnv(gym.Env):

    def __init__(self):

        # Create the observation space
        img_width = TRAINING_IMAGE_SIZE[0]
        img_height = TRAINING_IMAGE_SIZE[1]
        self.observation_space = spaces.Box(low=0, high=255, shape=(img_height, img_width, 3), dtype=np.uint8)

        # Create the action space
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([+1, +1]), dtype=np.float32)

        if node_type == SIMULATION_WORKER:

            # ROS initialization
            rospy.init_node('rl_coach', anonymous=True)
            rospy.Subscriber('/camera/zed/rgb/image_rect_color', sensor_image, self.callback_image)
            self.ack_publisher = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output',
                                                 AckermannDriveStamped, queue_size=100)
            self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)

            # Read in parameters
            self.world_name = rospy.get_param('WORLD_NAME')
            self.job_type = rospy.get_param('JOB_TYPE')
            self.aws_region = rospy.get_param('AWS_REGION')
            self.metrics_s3_bucket = rospy.get_param('METRICS_S3_BUCKET')
            self.metrics_s3_object_key = rospy.get_param('METRICS_S3_OBJECT_KEY')
            self.metrics = list()
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

            # Read in the waypoints
            BUNDLE_CURRENT_PREFIX = os.environ.get("BUNDLE_CURRENT_PREFIX", None)
            if not BUNDLE_CURRENT_PREFIX:
                raise ValueError("Cannot get BUNDLE_CURRENT_PREFIX")
            route_file_name = os.path.join(BUNDLE_CURRENT_PREFIX,
                'opt', 'install', 'deepracer_simulation_environment', 'share',
                'deepracer_simulation_environment', 'routes', '{}.npy'.format(self.world_name))
            waypoints = np.load(route_file_name)
            self.is_loop = np.all(waypoints[0,:] == waypoints[-1,:])
            if self.is_loop:
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

            # Initialize state data
            self.episodes = 0
            self.start_dist = 0.0
            self.round_robin = (self.job_type == TRAINING_JOB)
            self.is_simulation_done = False
            self.image = None
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

    def reset(self):
        if node_type == SAGEMAKER_TRAINING_WORKER:
            return self.observation_space.sample()

        # Simulation is done - so RoboMaker will start to shut down the app.
        # Till RoboMaker shuts down the app, do nothing more else metrics may show unexpected data.
        if (node_type == SIMULATION_WORKER) and self.is_simulation_done:
            while True:
                time.sleep(1)

        self.image = None
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
        self.send_action(0, 0)
        self.racecar_reset()
        time.sleep(SLEEP_AFTER_RESET_TIME_IN_SECOND)
        self.steps = 0
        self.simulation_start_time = time.time()

        # Compute the initial state
        self.infer_reward_state(0, 0)
        return self.next_state

    def racecar_reset(self):
        rospy.wait_for_service('/gazebo/set_model_state')

        # Compute the starting position and heading
        next_point_index = bisect.bisect(self.center_dists, self.start_dist)
        start_point = self.center_line.interpolate(self.start_dist, normalized=True)
        start_yaw = math.atan2(
            self.center_line.coords[next_point_index][1] - start_point.y,
            self.center_line.coords[next_point_index][0] - start_point.x)
        start_quaternion = Rotation.from_euler('zyx', [start_yaw, 0, 0]).as_quat()

        # Construct the model state and send to Gazebo
        modelState = ModelState()
        modelState.model_name = 'racecar'
        modelState.pose.position.x = start_point.x
        modelState.pose.position.y = start_point.y
        modelState.pose.position.z = 0
        modelState.pose.orientation.x = start_quaternion[0]
        modelState.pose.orientation.y = start_quaternion[1]
        modelState.pose.orientation.z = start_quaternion[2]
        modelState.pose.orientation.w = start_quaternion[3]
        modelState.twist.linear.x = 0
        modelState.twist.linear.y = 0
        modelState.twist.linear.z = 0
        modelState.twist.angular.x = 0
        modelState.twist.angular.y = 0
        modelState.twist.angular.z = 0
        self.set_model_state(modelState)

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
        self.send_action(self.steering_angle, self.speed)
        time.sleep(SLEEP_BETWEEN_ACTION_AND_REWARD_CALCULATION_TIME_IN_SECOND)
        self.steps += 1

        # Compute the next state and reward
        self.infer_reward_state(self.steering_angle, self.speed)
        return self.next_state, self.reward, self.done, {}

    def callback_image(self, data):
        self.image = data

    def send_action(self, steering_angle, speed):
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = rospy.Time.now()
        ack_msg.drive.steering_angle = steering_angle
        ack_msg.drive.speed = speed
        self.ack_publisher.publish(ack_msg)

    def infer_reward_state(self, steering_angle, speed):
        rospy.wait_for_service('/gazebo/get_model_state')
        rospy.wait_for_service('/gazebo/get_link_state')

        # Wait till we have a image from the camera
        while not self.image:
            time.sleep(SLEEP_WAITING_FOR_IMAGE_TIME_IN_SECOND)

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
        model_yaw = model_orientation.as_euler('zyx')[0]

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
        current_dist = self.center_line.project(model_point, normalized=True)
        next_waypoint_index = max(0, min(bisect.bisect(self.center_dists, current_dist), len(self.center_dists) - 1))
        prev_waypoint_index = next_waypoint_index - 1
        distance_from_next = model_point.distance(Point(self.center_line.coords[next_waypoint_index]))
        distance_from_prev = model_point.distance(Point(self.center_line.coords[prev_waypoint_index]))
        closest_waypoint_index = (prev_waypoint_index, next_waypoint_index)[distance_from_next < distance_from_prev]

        # Compute distance from center and road width
        nearest_point_center = self.center_line.interpolate(current_dist, normalized=True)
        nearest_point_inner = self.inner_border.interpolate(self.inner_border.project(nearest_point_center))
        nearest_point_outer = self.outer_border.interpolate(self.outer_border.project(nearest_point_center))
        distance_from_center = nearest_point_center.distance(model_point)
        road_width = nearest_point_inner.distance(nearest_point_outer)

        # Convert current progress to be [0,100] starting at the initial waypoint
        current_progress = current_dist - self.start_dist
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
            reward = self.reward_function(
                all_wheels_on_track, model_point.x, model_point.y, distance_from_center, model_yaw,
                current_progress, self.steps, speed, steering_angle, road_width,
                list(self.center_line.coords), closest_waypoint_index)
        else:
            done = True
            reward = CRASHED

        # Reset if the car position hasn't changed in the last 2 steps
        if min(model_point.distance(self.prev_point), model_point.distance(self.prev_point_2)) <= 0.0001:
            done = True
            reward = CRASHED  # stuck

        # Evaluations are done when progress reaches 100
        if self.job_type == EVALUATION_JOB:
            if current_progress >= 100:
                done = True

        # Keep data from the previous step around
        self.prev_point_2 = self.prev_point
        self.prev_point = model_point
        self.prev_progress = current_progress

        # Read the image and resize to get the state
        image = Image.frombytes('RGB', (self.image.width, self.image.height), self.image.data, 'raw', 'RGB', 0, 1)
        image = image.resize(TRAINING_IMAGE_SIZE, resample=2)
        state = np.array(image)

        # Set the next state, reward, and done flag
        self.next_state = state
        self.reward = reward
        self.reward_in_episode += reward
        self.done = done

        # Trace logs to help us debug and visualize the training runs
        # btown TODO: This should be written to S3, not to CWL.
        stdout_ = 'SIM_TRACE_LOG:%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%d,%.4f,%s,%s,%.4f,%d,%.2f,%s\n' % (
            self.episodes, self.steps, model_location[0], model_location[1], model_yaw,
            self.steering_angle,
            self.speed,
            self.action_taken,
            self.reward,
            self.done,
            all_wheels_on_track,
            current_progress,
            closest_waypoint_index,
            self.track_length,
            time.time())
        print(stdout_)

        # Terminate this episode when ready
        if self.done and node_type == SIMULATION_WORKER:
            self.finish_episode(current_progress)

    def finish_episode(self, progress):

        # Stop the car from moving
        self.send_action(0, 0)

        # Increment episode count, update start dist for round robin
        self.episodes += 1
        if self.round_robin:
            self.start_dist = 0.0 #(self.start_dist + ROUND_ROBIN_ADVANCE_DIST) % 1.0

        # Update metrics based on job type
        if self.job_type == TRAINING_JOB:
            #self.send_reward_to_cloudwatch(self.reward_in_episode)
            self.update_training_metrics()
            self.write_metrics_to_s3()
            if self.is_training_done():
                self.cancel_simulation_job()
        elif self.job_type == EVALUATION_JOB:
            self.number_of_trials += 1
            self.update_eval_metrics(progress)
            self.write_metrics_to_s3()
            if self.is_evaluation_done():
                self.cancel_simulation_job()

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

    def is_evaluation_done(self):
        if ((self.target_number_of_trials > 0) and (self.target_number_of_trials == self.number_of_trials)):
            self.is_simulation_done = True
        return self.is_simulation_done

    def is_training_done(self):
        if ((self.target_number_of_episodes > 0) and (self.target_number_of_episodes == self.episodes)) or \
           ((self.is_number(self.target_reward_score)) and (self.target_reward_score <= self.reward_in_episode)):
            self.is_simulation_done = True
        return self.is_simulation_done

    def is_number(self, value_to_check):
        try:
            float(value_to_check)
            return True
        except ValueError:
            return False

    def cancel_simulation_job(self):
        self.send_action(0, 0)
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
            with open('custom_files/model_metadata.json', 'r') as f:
                model_metadata = json.load(f)
                self.json_actions = model_metadata['action_space']
            logger.info("Loaded action space from file: {}".format(self.json_actions))
        except:
            # Failed to load, fall back on the default action space
            from markov.defaults import model_metadata
            self.json_actions = model_metadata['action_space']
            logger.info("Loaded default action space: {}".format(self.json_actions))
        self.action_space = spaces.Discrete(len(self.json_actions))

    def step(self, action):
        self.steering_angle = float(self.json_actions[action]['steering_angle']) * math.pi / 180.0;
        self.speed = float(self.json_actions[action]['speed'])
        self.action_taken = action
        return super().step([self.steering_angle, self.speed])
