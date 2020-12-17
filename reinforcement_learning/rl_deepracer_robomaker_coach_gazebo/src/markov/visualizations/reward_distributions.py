import rospy
from deepracer_simulation_environment.msg import AgentRewardData

class RewardDataPublisher(object):
    def __init__(self, agent_name, json_actions):
        print('Reward Distribution Graph: ', agent_name, json_actions)
        self.steering_angle_list = list()
        self.speed_list = list()
        self.action_space_len = len(json_actions)
        for json_action in json_actions:
            self.speed_list.append(str(json_action['speed']))
            self.steering_angle_list.append(str(json_action['steering_angle']))
        self.agent_name = agent_name
        self.reward_distribution_graph_publisher = rospy.Publisher('/reward_data/'+self.agent_name, AgentRewardData, queue_size=1)

    def publish_frame(self, image, action_index, reward_value):
        reward_data = AgentRewardData()
        reward_data.agent_name = self.agent_name
        reward_data.action = int(action_index)
        reward_data.reward = float(reward_value)
        reward_data.action_space_len = self.action_space_len
        reward_data.speed_list = self.speed_list
        reward_data.steering_angle_list = self.steering_angle_list
        reward_data.image = image
        self.reward_distribution_graph_publisher.publish(reward_data)

