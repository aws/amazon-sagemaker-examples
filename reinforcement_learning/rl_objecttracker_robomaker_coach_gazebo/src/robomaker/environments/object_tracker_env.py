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
import random
import math
import sys

TRAINING_IMAGE_SIZE = (160, 120)

# REWARD ENUM
CRASHED = 0
NO_PROGRESS = -1
REWARD_CONSTANT = 10.0
MAX_STEPS = 1000000

# SLEEP INTERVALS
SLEEP_AFTER_RESET_TIME_IN_SECOND = 0.5
SLEEP_BETWEEN_ACTION_AND_REWARD_CALCULATION_TIME_IN_SECOND = 0.1
SLEEP_WAITING_FOR_IMAGE_TIME_IN_SECOND = 0.01

# Type of worker
SIMULATION_WORKER = "SIMULATION_WORKER"
SAGEMAKER_TRAINING_WORKER = "SAGEMAKER_TRAINING_WORKER"

node_type = os.environ.get("NODE_TYPE", SIMULATION_WORKER)

if node_type == SIMULATION_WORKER:
    import rospy
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image as sensor_image
    from gazebo_msgs.srv import SetModelState
    from gazebo_msgs.msg import ModelState

### Gym Env ###
class TurtleBot3ObjectTrackerAndFollowerEnv(gym.Env):
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
        self.burger_x = 0
        self.burger_y = 0

        # actions -> steering angle, throttle
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([+1, +1]), dtype=np.float32)

        # given image from simulator
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(screen_height, screen_width, 3), dtype=np.uint8)

        if node_type == SIMULATION_WORKER:
            #ROS initialization
            self.ack_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=100)
            self.gazebo_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            rospy.init_node('rl_coach', anonymous=True)

            #Subscribe to ROS topics and register callbacks
            rospy.Subscriber('/odom', Odometry, self.callback_position)
            rospy.Subscriber('/camera/rgb/image_raw', sensor_image, self.callback_image)
            self.aws_region = rospy.get_param('ROS_AWS_REGION')

        self.reward_in_episode = 0
        self.steps = 0
        self.last_distance_of_turtlebot = sys.maxsize

    def reset(self):
        if node_type == SAGEMAKER_TRAINING_WORKER:
            return self.observation_space.sample()
        print('Total Reward Reward=%.2f' % self.reward_in_episode,
              'Total Steps=%.2f' % self.steps)
        self.send_reward_to_cloudwatch(self.reward_in_episode)

        self.reward = None
        self.done = False
        self.next_state = None
        self.image = None
        self.steps = 0
        self.prev_progress = 0
        self.reward_in_episode = 0

        self.send_action(0, 0) # set the throttle to 0
        self.turtlebot3_reset()

        self.infer_reward_state()
        return self.next_state

    def turtlebot3_reset(self):
        rospy.wait_for_service('gazebo/set_model_state')

        self.x = 0
        self.y = 0

        # Put the turtlebot waffle at (0, 0)
        modelState = ModelState()
        modelState.pose.position.z = 0
        modelState.pose.orientation.x = 0
        modelState.pose.orientation.y = 0
        modelState.pose.orientation.z = 0
        modelState.pose.orientation.w = 0
        modelState.twist.linear.x = 0
        modelState.twist.linear.y = 0
        modelState.twist.linear.z = 0
        modelState.twist.angular.x = 0
        modelState.twist.angular.y = 0
        modelState.twist.angular.z = 0
        modelState.model_name = 'turtlebot3'
        modelState.pose.position.x = self.x
        modelState.pose.position.y = self.y
        self.gazebo_model_state_service(modelState)

        self.burger_x = 3.5
        self.burger_y = random.uniform(-1, 1)

        # Put the turtlebot burger at (2, 0)
        modelState = ModelState()
        modelState.pose.position.z = 0
        modelState.pose.orientation.x = 0
        modelState.pose.orientation.y = 0
        modelState.pose.orientation.z = 0
        modelState.pose.orientation.w = random.uniform(0, 3)
        modelState.twist.linear.x = 0
        modelState.twist.linear.y = 0
        modelState.twist.linear.z = 0
        modelState.twist.angular.x = 0
        modelState.twist.angular.y = 0
        modelState.twist.angular.z = 0
        modelState.model_name = 'turtlebot3_burger'
        modelState.pose.position.x = self.burger_x
        modelState.pose.position.y = self.burger_y
        self.gazebo_model_state_service(modelState)

        self.last_distance_of_turtlebot = sys.maxsize

        time.sleep(SLEEP_AFTER_RESET_TIME_IN_SECOND)

    def step(self, action):
        if node_type == SAGEMAKER_TRAINING_WORKER:
            return self.observation_space.sample(), 0, False, {}

        #initialize rewards, next_state, done
        self.reward = None
        self.done = False
        self.next_state = None

        steering = float(action[0])
        throttle = float(action[1])
        self.steps += 1
        self.send_action(steering, throttle)
        time.sleep(SLEEP_BETWEEN_ACTION_AND_REWARD_CALCULATION_TIME_IN_SECOND)
        self.infer_reward_state()

        info = {} #additional data, not to be used for training
        return self.next_state, self.reward, self.done, info

    def callback_image(self, data):
        self.image = data

    def callback_position(self, data):
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y

    def send_action(self, steering, throttle):
        speed = Twist()
        speed.linear.x = throttle
        speed.angular.z = steering
        self.ack_publisher.publish(speed)

    def reward_function(self, distance_of_turtlebot):
        return REWARD_CONSTANT / (distance_of_turtlebot * distance_of_turtlebot)

    def infer_reward_state(self):
        #Wait till we have a image from the camera
        while not self.image:
            time.sleep(SLEEP_WAITING_FOR_IMAGE_TIME_IN_SECOND)

        image = Image.frombytes('RGB', (self.image.width, self.image.height),
                                self.image.data,'raw', 'BGR', 0, 1)
        image = image.resize(TRAINING_IMAGE_SIZE)
        state = np.array(image)

        x = self.x
        y = self.y

        distance_of_turtlebot = math.sqrt((x - self.burger_x) * (x - self.burger_x)
                                          + (y - self.burger_y) * (y - self.burger_y))
        done = False

        reward = 0

        if distance_of_turtlebot < self.last_distance_of_turtlebot:
            self.last_distance_of_turtlebot = distance_of_turtlebot
            reward = self.reward_function(distance_of_turtlebot)
            if distance_of_turtlebot < 0.2:
                done = True

        if distance_of_turtlebot > 5:
            done = True

        self.reward_in_episode += reward
        print('Step No=%.2f' % self.steps,
              'Reward=%.2f' % reward,
              'Distance of bot=%f' % distance_of_turtlebot)

        self.reward = reward
        self.done = done
        self.next_state = state

    def send_reward_to_cloudwatch(self, reward):
        session = boto3.session.Session()
        cloudwatch_client = session.client('cloudwatch', region_name=self.aws_region)
        cloudwatch_client.put_metric_data(
            MetricData=[
                {
                    'MetricName': 'ObjectTrackerRewardPerEpisode',
                    'Unit': 'None',
                    'Value': reward
                },
            ],
            Namespace='AWSRoboMakerSimulation'
        )

class TurtleBot3ObjectTrackerAndFollowerDiscreteEnv(TurtleBot3ObjectTrackerAndFollowerEnv):
    def __init__(self):
        TurtleBot3ObjectTrackerAndFollowerEnv.__init__(self)

        # actions -> straight, left, right
        self.action_space = spaces.Discrete(5)

    def step(self, action):

        # Convert discrete to continuous
        if action == 0:  # move left
            steering = 0.6
            throttle = 0.1
        elif action == 1:  # move right
            steering = -0.6
            throttle = 0.1
        elif action == 2:  # straight
            steering = 0
            throttle = 0.1
        elif action == 3:  # move left
            steering = 0.3
            throttle = 0.1
        elif action == 4:  # move right
            steering = -0.3
            throttle = 0.1
        else:  # should not be here
            raise ValueError("Invalid action")

        continous_action = [steering, throttle]

        return super().step(continous_action)