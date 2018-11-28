from __future__ import print_function

import time

# only needed for fake driver setup
import boto3
# gym
import gym
import numpy as np
from gym import spaces
from PIL import Image
import os
import math

# Type of worker
SIMULATION_WORKER = "SIMULATION_WORKER"
SAGEMAKER_TRAINING_WORKER = "SAGEMAKER_TRAINING_WORKER"

node_type = os.environ.get("NODE_TYPE", SIMULATION_WORKER)

if node_type == SIMULATION_WORKER:
    import rospy
    from ackermann_msgs.msg import AckermannDriveStamped
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import SetModelState

    from sensor_msgs.msg import Image as sensor_image
    from deepracer_msgs.msg import Progress

TRAINING_IMAGE_SIZE = (160, 120)
FINISH_LINE = 100

# REWARD ENUM
CRASHED = 0
NO_PROGRESS = -1
FINISHED = 10000000.0
MAX_STEPS = 1000000

# WORLD NAME
EASY_TRACK_WORLD = 'easy_track'
MEDIUM_TRACK_WORLD = 'medium_track'
HARD_TRACK_WORLD = 'hard_track'

# SLEEP INTERVALS
SLEEP_AFTER_RESET_TIME_IN_SECOND = 0.5
SLEEP_BETWEEN_ACTION_AND_REWARD_CALCULATION_TIME_IN_SECOND = 0.1
SLEEP_WAITING_FOR_IMAGE_TIME_IN_SECOND = 0.01

### Gym Env ###
class DeepRacerEnv(gym.Env):
    def __init__(self):

        screen_height = TRAINING_IMAGE_SIZE[1]
        screen_width = TRAINING_IMAGE_SIZE[0]

        self.on_track = 0
        self.progress = 0
        self.yaw = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.distance_from_center = 0
        self.distance_from_border_1 = 0
        self.distance_from_border_2 = 0
        self.steps = 0
        self.progress_at_beginning_of_race = 0

        # actions -> steering angle, throttle
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([+1, +1]), dtype=np.float32)

        # given image from simulator
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(screen_height, screen_width, 3), dtype=np.uint8)

        if node_type == SIMULATION_WORKER:
            # ROS initialization
            self.ack_publisher = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output',
                                                 AckermannDriveStamped, queue_size=100)
            self.racecar_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            rospy.init_node('rl_coach', anonymous=True)

            # Subscribe to ROS topics and register callbacks
            rospy.Subscriber('/progress', Progress, self.callback_progress)
            rospy.Subscriber('/camera/zed/rgb/image_rect_color', sensor_image, self.callback_image)
            self.world_name = rospy.get_param('WORLD_NAME')
            self.set_waypoints()
            self.aws_region = rospy.get_param('ROS_AWS_REGION')

        self.reward_in_episode = 0
        self.prev_progress = 0
        self.steps = 0

    def reset(self):
        if node_type == SAGEMAKER_TRAINING_WORKER:
            return self.observation_space.sample()
        print('Total Reward Reward=%.2f' % self.reward_in_episode,
              'Total Steps=%.2f' % self.steps)
        self.send_reward_to_cloudwatch(self.reward_in_episode)

        self.reward_in_episode = 0
        self.reward = None
        self.done = False
        self.next_state = None
        self.image = None
        self.steps = 0
        self.prev_progress = 0

        # Reset car in Gazebo
        self.send_action(0, 0)  # set the throttle to 0
        self.racecar_reset()

        self.infer_reward_state(0, 0)
        return self.next_state

    def racecar_reset(self):
        rospy.wait_for_service('gazebo/set_model_state')

        modelState = ModelState()
        modelState.pose.position.z = 0
        modelState.pose.orientation.x = 0
        modelState.pose.orientation.y = 0
        modelState.pose.orientation.z = 0
        modelState.pose.orientation.w = 0  # Use this to randomize the orientation of the car
        modelState.twist.linear.x = 0
        modelState.twist.linear.y = 0
        modelState.twist.linear.z = 0
        modelState.twist.angular.x = 0
        modelState.twist.angular.y = 0
        modelState.twist.angular.z = 0
        modelState.model_name = 'racecar'

        if self.world_name.startswith(MEDIUM_TRACK_WORLD):
            modelState.pose.position.x = -1.40
            modelState.pose.position.y = 2.13
        elif self.world_name.startswith(EASY_TRACK_WORLD):
            modelState.pose.position.x = -1.44
            modelState.pose.position.y = -0.06
        elif self.world_name.startswith(HARD_TRACK_WORLD):
            modelState.pose.position.x = 1.75
            modelState.pose.position.y = 0.6
        else:
            raise ValueError("Unknown simulation world: {}".format(self.world_name))

        self.racecar_service(modelState)
        time.sleep(SLEEP_AFTER_RESET_TIME_IN_SECOND)
        self.progress_at_beginning_of_race = self.progress

    def step(self, action):
        if node_type == SAGEMAKER_TRAINING_WORKER:
            return self.observation_space.sample(), 0, False, {}

        # initialize rewards, next_state, done
        self.reward = None
        self.done = False
        self.next_state = None

        steering_angle = float(action[0])
        throttle = float(action[1])
        self.steps += 1
        self.send_action(steering_angle, throttle)
        time.sleep(SLEEP_BETWEEN_ACTION_AND_REWARD_CALCULATION_TIME_IN_SECOND)
        self.infer_reward_state(steering_angle, throttle)

        info = {}  # additional data, not to be used for training
        return self.next_state, self.reward, self.done, info

    def callback_image(self, data):
        self.image = data

    def callback_progress(self, data):
        self.on_track = not (data.off_track)
        self.progress = data.progress
        self.yaw = data.yaw
        self.x = data.x
        self.y = data.y
        self.z = data.z
        self.distance_from_center = data.distance_from_center
        self.distance_from_border_1 = data.distance_from_border_1
        self.distance_from_border_2 = data.distance_from_border_2

    def send_action(self, steering_angle, throttle):
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = rospy.Time.now()
        ack_msg.drive.steering_angle = steering_angle
        ack_msg.drive.speed = throttle
        self.ack_publisher.publish(ack_msg)

    def reward_function(self, on_track, x, y, distance_from_center, car_orientation, progress, steps,
                        throttle, steering, track_width, waypoints, closest_waypoints):
        if distance_from_center >= 0.0 and distance_from_center <= 0.02:
            return 1.0
        elif distance_from_center >= 0.02 and distance_from_center <= 0.03:
            return 0.3
        elif distance_from_center >= 0.03 and distance_from_center <= 0.05:
            return 0.1
        return 1e-3  # like crashed

    def infer_reward_state(self, steering_angle, throttle):
        # Wait till we have a image from the camera
        while not self.image:
            time.sleep(SLEEP_WAITING_FOR_IMAGE_TIME_IN_SECOND)

        # Car environment spits out BGR images by default. Converting to the
        # image to RGB.
        image = Image.frombytes('RGB', (self.image.width, self.image.height),
                                self.image.data, 'raw', 'BGR', 0, 1)
        # resize image ans perform anti-aliasing
        image = image.resize(TRAINING_IMAGE_SIZE, resample=2).convert("RGB")
        state = np.array(image)

        on_track = self.on_track
        total_progress = self.progress - self.progress_at_beginning_of_race
        done = False

        self.prev_progress = total_progress

        if on_track != 1:
            reward = CRASHED
            done = True
        elif total_progress >= FINISH_LINE:  # reached max waypoints
            print("Congratulations! You finished the race!")
            if self.steps == 0:
                reward = 0.0
                done = False
            else:
                reward = FINISHED / self.steps
                done = True
        else:
            reward = self.reward_function(on_track, self.x, self.y, self.distance_from_center, self.yaw,
                                          total_progress, self.steps, throttle, steering_angle, self.road_width,
                                          list(self.waypoints), self.get_closest_waypoint())

        print('Step No=%.2f' % self.steps,
              'Step Reward=%.2f' % reward)

        self.reward_in_episode += reward
        self.reward = reward
        self.done = done
        self.next_state = state

    def send_reward_to_cloudwatch(self, reward):
        session = boto3.session.Session()
        cloudwatch_client = session.client('cloudwatch', region_name=self.aws_region)
        cloudwatch_client.put_metric_data(
            MetricData=[
                {
                    'MetricName': 'DeepRacerRewardPerEpisode',
                    'Unit': 'None',
                    'Value': reward
                },
            ],
            Namespace='AWSRoboMakerSimulation'
        )

    def set_waypoints(self):
        if self.world_name.startswith(MEDIUM_TRACK_WORLD):
            self.waypoints = vertices = np.zeros((8, 2))
            self.road_width = 0.50
            vertices[0][0] = -0.99; vertices[0][1] = 2.25;
            vertices[1][0] = 0.69;  vertices[1][1] = 2.26;
            vertices[2][0] = 1.37;  vertices[2][1] = 1.67;
            vertices[3][0] = 1.48;  vertices[3][1] = -1.54;
            vertices[4][0] = 0.81;  vertices[4][1] = -2.44;
            vertices[5][0] = -1.25; vertices[5][1] = -2.30;
            vertices[6][0] = -1.67; vertices[6][1] = -1.64;
            vertices[7][0] = -1.73; vertices[7][1] = 1.63;
        elif self.world_name.startswith(EASY_TRACK_WORLD):
            self.waypoints = vertices = np.zeros((2, 2))
            self.road_width = 0.90
            vertices[0][0] = -1.08;   vertices[0][1] = -0.05;
            vertices[1][0] =  1.08;   vertices[1][1] = -0.05;
        else:
            self.waypoints = vertices = np.zeros((30, 2))
            self.road_width = 0.44
            vertices[0][0] = 1.5;     vertices[0][1] = 0.58;
            vertices[1][0] = 5.5;     vertices[1][1] = 0.58;
            vertices[2][0] = 5.6;     vertices[2][1] = 0.6;
            vertices[3][0] = 5.7;     vertices[3][1] = 0.65;
            vertices[4][0] = 5.8;     vertices[4][1] = 0.7;
            vertices[5][0] = 5.9;     vertices[5][1] = 0.8;
            vertices[6][0] = 6.0;     vertices[6][1] = 0.9;
            vertices[7][0] = 6.08;    vertices[7][1] = 1.1;
            vertices[8][0] = 6.1;     vertices[8][1] = 1.2;
            vertices[9][0] = 6.1;     vertices[9][1] = 1.3;
            vertices[10][0] = 6.1;    vertices[10][1] = 1.4;
            vertices[11][0] = 6.07;   vertices[11][1] = 1.5;
            vertices[12][0] = 6.05;   vertices[12][1] = 1.6;
            vertices[13][0] = 6;      vertices[13][1] = 1.7;
            vertices[14][0] = 5.9;    vertices[14][1] = 1.8;
            vertices[15][0] = 5.75;   vertices[15][1] = 1.9;
            vertices[16][0] = 5.6;    vertices[16][1] = 2.0;
            vertices[17][0] = 4.2;    vertices[17][1] = 2.02;
            vertices[18][0] = 4;      vertices[18][1] = 2.1;
            vertices[19][0] = 2.6;    vertices[19][1] = 3.92;
            vertices[20][0] = 2.4;    vertices[20][1] = 4;
            vertices[21][0] = 1.2;    vertices[21][1] = 3.95;
            vertices[22][0] = 1.1;    vertices[22][1] = 3.92;
            vertices[23][0] = 1;      vertices[23][1] = 3.88;
            vertices[24][0] = 0.8;    vertices[24][1] = 3.72;
            vertices[25][0] = 0.6;    vertices[25][1] = 3.4;
            vertices[26][0] = 0.58;   vertices[26][1] = 3.3;
            vertices[27][0] = 0.57;   vertices[27][1] = 3.2;
            vertices[28][0] = 1;      vertices[28][1] = 1;
            vertices[29][0] = 1.25;   vertices[29][1] = 0.7;

    def get_closest_waypoint(self):
        res = 0
        index = 0
        x = self.x
        y = self.y
        minDistance = float('inf')
        for row in self.waypoints:
            distance = math.sqrt((row[0] - x) * (row[0] - x) + (row[1] - y) * (row[1] - y))
            if distance < minDistance:
                minDistance = distance
                res = index
            index = index + 1
        return res

class DeepRacerDiscreteEnv(DeepRacerEnv):
    def __init__(self):
        DeepRacerEnv.__init__(self)

        # actions -> straight, left, right
        self.action_space = spaces.Discrete(5)

    def step(self, action):

        # Convert discrete to continuous
        if action == 0:  # move left
            steering_angle = 0.8
            throttle = 0.3
        elif action == 1:  # move right
            steering_angle = -0.8  # -1 #-0.5 #-1
            throttle = 0.3
        elif action == 2:  # straight
            steering_angle = 0
            throttle = 0.3
        elif action == 3:  # move left
            steering_angle = 0.4
            throttle = 0.3
        elif action == 4:  # move right
            steering_angle = -0.4  # -1 #-0.5 #-1
            throttle = 0.3
        else:  # should not be here
            raise ValueError("Invalid action")

        continous_action = [steering_angle, throttle]

        return super().step(continous_action)