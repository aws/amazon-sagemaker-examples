import gym
import numpy as np
import pandas as pd
import json
from pathlib import Path

gym.logger.set_level(40)

class VectoredGymEnvironment():
    """
    Envrioment class to run multiple similations and collect rollout data
    """
    def __init__(self, registered_gym_env, num_of_envs=1):
        self.envs_initialized = False
        self.initialized_envs = {}
        self.env_states = {}
        self.env_reset_counter = {}
        self.num_of_envs = num_of_envs
        self.data_rows = []

        self.initialize_envs(num_of_envs, registered_gym_env)
 
    def is_initialized(self):
        return self.envs_initialized
 
    def initialize_envs(
            self,
            num_of_envs,
            registered_gym_env):
        """Initialize multiple Openai gym environments.
        Each envrionment will start with a different random seed.

        Arguments:
            num_of_envs {int} -- Number of environments/simulations to initiate
            registered_gym_env {str} -- Environment name of the registered gym environment
        """
        print("Initializing {} environments of {}".format(num_of_envs, registered_gym_env))
        for i in range(0, num_of_envs):
            environment_id = "environment_" + str(i)
            environment = gym.make(registered_gym_env)
            environment = environment.unwrapped
            environment.seed(i)
            self.env_states[environment_id] = environment.reset()
            self.env_reset_counter[environment_id] = 0
            self.initialized_envs[environment_id] = environment
        self.envs_initialized = True
        self.state_dims = len(self.env_states[environment_id])

    def get_environment_states(self):
        return self.env_states

    def dump_environment_states(self, dir_path, file_name):
        """Dumping current states of all the envrionments into file
        
        Arguments:
            dir_path {str} -- Directory path of the target file
            file_name {str} -- File name of the target file
        """
        data_folder = Path(dir_path)
        file_path = data_folder / file_name

        with open(file_path, 'w') as outfile:
            for state in self.env_states.values():
                json.dump(list(state), outfile)
                outfile.write('\n')

    def get_environment_ids(self):
        return list(self.initialized_envs.keys())
 
    def step(self, environment_id, action):
        local_env = self.initialized_envs[environment_id]
        observation, reward, done, info = local_env.step(action)

        self.env_states[environment_id] = observation
        return observation, reward, done, info
 
    def reset(self, environment_id):
        self.env_states[environment_id] = \
            self.initialized_envs[environment_id].reset()
        return self.env_states[environment_id]

    def reset_all_envs(self):
        print("Resetting all the environments...")
        for i in range(0, self.num_of_envs): 
            environment_id = "environment_" + str(i)
            self.reset(environment_id)
 
    def close(self, environment_id):
        self.initialized_envs[environment_id].close()
        return
 
    def render(self, environment_id):
        self.initialized_envs[environment_id].render()
        return

    def collect_rollouts_for_single_env_with_given_episodes(self, environment_id, action_prob, num_episodes):
        """Collect rollouts with given steps from one environment
        
        Arguments:
            environment_id {str} -- Environment id for the environment
            action_prob {list} -- Action probabilities of the simulated policy
            num_episodes {int} -- Number of episodes to run rollouts
        """
        # normalization if sum of probs is not exact equal to 1
        action_prob = np.array(action_prob)
        if action_prob.sum() != 1:
            action_prob /= action_prob.sum()
        action_prob = list(action_prob)

        for _ in range(num_episodes):
            done = False
            cumulative_rewards = 0
            while not done:
                data_item = []
                action = np.random.choice(len(action_prob), p=action_prob)
                cur_state_features = self.env_states[environment_id]
                _, reward, done, _ = self.step(environment_id, action)
                cumulative_rewards += reward
                episode_id = int(environment_id.split('_')[-1]) + \
                    self.num_of_envs * self.env_reset_counter[environment_id]
                if not done:
                    data_item.extend([action, action_prob, episode_id, reward, 0.0])
                else:
                    data_item.extend([action, action_prob, episode_id, reward, cumulative_rewards])
                for j in range(len(cur_state_features)):
                    data_item.append(cur_state_features[j])
                self.data_rows.append(data_item)

            self.reset(environment_id)
            self.env_reset_counter[environment_id] += 1

    def collect_rollouts_for_single_env_with_given_steps(self, environment_id, action_prob, num_steps):
        """Collect rollouts with given steps from one environment
        
        Arguments:
            environment_id {str} -- Environment id for the environment
            action_prob {list} -- Action probabilities of the simulated policy
            num_episodes {int} -- Number of steps to run rollouts
        """
        # normalization if sum of probs is not exact equal to 1
        action_prob = np.array(action_prob)
        if action_prob.sum() != 1:
            action_prob /= action_prob.sum()
        action_prob = list(action_prob)

        for _ in range(num_steps):
            data_item = []
            action = np.random.choice(len(action_prob), p=action_prob)
            cur_state_features = self.env_states[environment_id]
            _, reward, done, _ = self.step(environment_id, action)
            episode_id = int(environment_id.split('_')[-1]) + \
                self.num_of_envs * self.env_reset_counter[environment_id]
            data_item.extend([action, action_prob, episode_id, reward])
            for j in range(len(cur_state_features)):
                data_item.append(cur_state_features[j])
            self.data_rows.append(data_item)
            if done:
                self.reset(environment_id)
                self.env_reset_counter[environment_id] += 1

    def collect_rollouts_with_given_action_probs(self, num_steps=None, num_episodes=None, action_probs=None, file_name=None):
        """Collect rollouts from all the initiated environments with given action probs
        
        Keyword Arguments:
            num_steps {int} -- Number of steps to run rollouts (default: {None})
            num_episodes {int} --  Number of episodes to run rollouts (default: {None})
            action_probs {list} -- Action probs for the policy (default: {None})
            file_name {str} -- Batch transform output that contain predictions of probs  (default: {None})
        
        Returns:
            [Dataframe] -- Dataframe that contains the rollout data from all envs
        """
        if file_name is not None:
            assert action_probs is None
            json_lines = [json.loads(line.rstrip('\n')) for line in open(file_name) if line is not '']
            action_probs = []
            for line in json_lines:
                if line.get('SageMakerOutput') is not None:
                    action_probs.append(line['SageMakerOutput'].get("predictions")[0])
                else:
                    action_probs.append(line.get("predictions")[0])

        assert len(action_probs) == self.num_of_envs
        for index, environment_id in enumerate(self.get_environment_ids()):
            if num_steps is not None:
                assert num_episodes is None
                self.collect_rollouts_for_single_env_with_given_steps(
                    environment_id, action_probs[index], num_steps
                )
            else:
                assert num_episodes is not None
                self.collect_rollouts_for_single_env_with_given_episodes(
                    environment_id, action_probs[index], num_episodes
                )

        col_names = self._create_col_names()
        df = pd.DataFrame(self.data_rows, columns = col_names)

        return df

    def _create_col_names(self):
        """Create column names of dataframe that can be consumed by Coach
        
        Returns:
            [list] -- List of column names
        """
        col_names = ['action', 'all_action_probabilities', 'episode_id', 'reward', 'cumulative_rewards']
        for i in range(self.state_dims):
            col_names.append('state_feature_' + str(i))

        return col_names