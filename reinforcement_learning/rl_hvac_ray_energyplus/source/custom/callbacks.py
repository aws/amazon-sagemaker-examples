from typing import Dict

import numpy as np
import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class CustomCallbacks(DefaultCallbacks):
    """
    Please refer to :
        https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
        https://docs.ray.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
    for examples on adding your custom metrics and callbacks.

    This code adapts the documentations of the individual functions from :
    https://github.com/ray-project/ray/blob/master/rllib/agents/callbacks.py

    These callbacks can be used for custom metrics and custom postprocessing.
    """

    def on_episode_start(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        **kwargs
    ):
        """Callback run on the rollout worker before each episode starts.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        pass

    def on_episode_step(
        self, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, **kwargs
    ):
        """Runs on each episode step.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        pass

    def on_episode_end(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        **kwargs
    ):
        """Runs when an episode is done.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        ######################################################################
        # An example of adding a custom metric from the latest observation
        # from your env
        ######################################################################
        # last_obs_object_from_episode = episode.last_observation_for()
        # We define a dummy custom metric, observation_mean
        # episode.custom_metrics["observation_mean"] = last_obs_object_from_episode.mean()
        pass

    def on_postprocess_trajectory(
        self,
        worker: RolloutWorker,
        episode: MultiAgentEpisode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, SampleBatch],
        **kwargs
    ):
        """Called immediately after a policy's postprocess_fn is called.
        You can use this callback to do additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            episode (MultiAgentEpisode): Episode object.
            agent_id (str): Id of the current agent.
            policy_id (str): Id of the current policy for the agent.
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            postprocessed_batch (SampleBatch): The postprocessed sample batch
                for this agent. You can mutate this object to apply your own
                trajectory postprocessing.
            original_batches (dict): Mapping of agents to their unpostprocessed
                trajectory data. You should not mutate this object.
            kwargs: Forward compatibility placeholder.
        """
        pass

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        """Called at the end RolloutWorker.sample().
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            samples (SampleBatch): Batch to be returned. You can mutate this
                object to modify the samples generated.
            kwargs: Forward compatibility placeholder.
        """
        pass

    def on_train_result(self, trainer, result: dict, **kwargs):
        """Called at the end of Trainable.train().
        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # In this case we also print the mean timesteps throughput
        # for easier reference in the logs
        # print("=============================================================")
        # print(" Timesteps Throughput : {} ts/sec".format(TBD))
        # print("=============================================================")
        pass
