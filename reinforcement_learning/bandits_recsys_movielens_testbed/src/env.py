import os
import numpy as np


class MovieLens100KEnv():
    def __init__(self, data_dir="./ml-100k", item_pool_size=None, top_k=5, max_users=100):
        """
        Args:
            data_dir: Local dir where MovieLens 100K has been extracted.

            item_pool_size: The size of candidate list - environment will randomly select
            these many items for the movies that the user has rated in the dataset. If None,
            environment will use all the movies rated by the user.

            top_k: The size of the slate or the no. of items that the agent will recommend. The environment
            needs this to calculate the optimal expected reward.

            max_users: The environment will sample from `max_users` only. If set to None,
            all users i.e. 943 will be used for sampling. This parameter can be used to 
            simplify the learning problem.

        """
        self._preprocess_data(data_dir)
        self.total_users, self.total_items = self.attractiveness_means.shape
        self.max_users = max_users
        self.item_pool_size = item_pool_size
        self.top_k = top_k
        self._reset()

    def _preprocess_data(self, data_dir):
        metadata_file = os.path.join(data_dir, 'u.item')
        genre_file = os.path.join(data_dir, 'u.genre')
        ratings_data = os.path.join(data_dir, 'u.data')

        num_users = 943
        num_items = 1682

        self.attractiveness_means = np.zeros((num_users, num_items))
        self.item_features = np.zeros((num_items, 19))
        movie_names = {}

        with open(metadata_file, encoding='latin-1') as f:
            for line in f.readlines():
                line = line.strip().split("|")
                item_id = int(line[0]) - 1
                movie_names[item_id] = line[1]
                self.item_features[item_id][:] = list(map(int, line[5:]))

        with open(ratings_data) as f:
            for line in f.readlines():
                line = line.strip().split()
                user_id = int(line[0]) - 1
                item_id = int(line[1]) - 1
                rating = float(line[2])
                if rating >= 3:
                    rating = rating / 5
                else:
                    rating = 0.01
                self.attractiveness_means[user_id][item_id] = rating

    def _reset(self):
        self.done = False
        self.current_user_id = None
        self.current_user_embedding = None
        self.current_item_pool = None
        self.current_items_embedding = None
        self.step_count = 0
        self.total_regret = 0
        self.total_random_regret = 0

    def reset(self):
        self._reset()
        self._regulate_item_pool()
        return self.current_user_embedding, self.current_items_embedding

    def _regulate_item_pool(self):
        if self.step_count > self.total_users - 1:
            self.step_count = 0

        if self.max_users:
            if self.step_count > self.max_users - 1:
                self.step_count = 0

        # TODO: Randomize user selection
        self.current_user_id = self.step_count
        self.current_user_embedding = None

        # List of all the items that the user has rated in the past
        self.current_item_pool = np.flatnonzero(self.attractiveness_means[self.current_user_id])
        if self.item_pool_size and (len(self.current_item_pool) > self.item_pool_size):
            random_indices = np.random.choice(len(self.current_item_pool), size=self.item_pool_size, replace=False)
            self.current_item_pool = self.current_item_pool[random_indices]
        self.current_items_embedding = self.item_features[self.current_item_pool]

    def step(self, actions):
        assert len(actions) == self.top_k, "Size of recommended items list does not match top-k"
        rewards, regret, random_regret = self.get_feedback(actions)
        self.total_regret += regret
        self.total_random_regret += random_regret
        info = {"total_regret": self.total_regret, "total_random_regret": self.total_random_regret}
        self.step_count += 1
        self._regulate_item_pool()
        return (self.current_user_embedding, self.current_items_embedding), rewards, False, info

    def get_feedback(self, actions, click_model="cascade"):
        """
        Return rewards: List[float] and regret for the current recommended list - actions

        Args:
            actions: A list of top-k actions indices picked by the agent from candidate list
            click_model: One of 'cascade', 'pbm'

        Returns:
            rewards: A reward corresponding to each item in the list
            regret: Expected regret calculated based on the recommended actions
            regret_random: Expected regret calculated based on the actions of a random agent
        """
        # TODO: Implement PBM: Position based model

        recommended_item_ids = self.current_item_pool[actions]
        attraction_probs = self.attractiveness_means[self.step_count][recommended_item_ids]
        
        random_indices = np.random.choice(len(recommended_item_ids), size=self.top_k, replace=False)
        random_item_ids = self.current_item_pool[random_indices]
        random_attraction_probs = self.attractiveness_means[self.step_count][random_item_ids]
        # Simulate user behavior using a cascading click model.
        # User scans the list top-down and clicks on an item with prob = attractiveness_means.
        # User stops seeing the list after the first click.

        clicks = np.random.binomial(1, attraction_probs)
        if clicks.sum() > 1:
            first_click = np.flatnonzero(clicks)[0]
            clicks = clicks[:first_click + 1]

        expected_reward = 1 - np.prod(1 - attraction_probs)
        expected_reward_random = 1 - np.prod(1 - random_attraction_probs)

        current_pool_probs = self.attractiveness_means[self.step_count][self.current_item_pool]
        optimal_attraction_probs = np.sort(current_pool_probs)[::-1][:self.top_k]
        expected_optimal_reward = 1 - np.prod(1 - optimal_attraction_probs)
        regret = expected_optimal_reward - expected_reward
        regret_random = expected_optimal_reward - expected_reward_random

        return clicks, regret, regret_random