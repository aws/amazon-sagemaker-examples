import gym
import roboschool
import os

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines.ppo1 import PPO1
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines.common import tf_util
from stable_baselines.common.policies import MlpPolicy
from mpi4py import MPI


class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, _reward):
        return _reward * self.scale


class SagemakerStableBaselinesLauncher():
    """
    Sagemaker's Stable Baselines Launcher.
    """

    def __init__(self, env, output_path, model, num_timesteps):
        self._env = env
        self._output_path = output_path
        self._model = model
        self._num_timesteps = num_timesteps

    def _train(self):
        """Train the RL model
        """
        self._model.learn(total_timesteps=self._num_timesteps)

    def _predict(self, model, video_path):
        """Run predictions on trained RL model.
        """

        vr = VideoRecorder(env=self._env, path="{}/rl_out.mp4".format(video_path, str(MPI.COMM_WORLD.Get_rank())),
                           enabled=True)
        obs = self._env.reset()
        for i in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = self._env.step(action)
            if dones:
                obs = self._env.reset()
            self._env.render(mode='rgb_array')
            vr.capture_frame()
        vr.close()
        self._env.close()

    def run(self):

        self._train()

        if MPI.COMM_WORLD.Get_rank() == 0:
            self._predict(self._model, self._output_path)


class SagemakerStableBaselinesPPO1Launcher(SagemakerStableBaselinesLauncher):
    """
    Sagemaker's Stable Baselines PPO1 Launcher.
    """

    def __init__(self, env, output_path, timesteps_per_actorbatch,
                 clip_param, entcoeff, optim_epochs,
                 optim_stepsize, optim_batchsize,
                 gamma, lam, schedule,
                 verbose, num_timesteps):
        print(
            "Initializing PPO with output_path: {} and Hyper Params [timesteps_per_actorbatch: {},clip_param: {}, "
            "entcoeff: {}, optim_epochs: {}, optim_stepsize: {}, optim_batchsize: {}, gamma: {}, lam: {}, "
            "schedule: {}, verbose: {}, num_timesteps: {}]".format(output_path, timesteps_per_actorbatch,
                                                                   clip_param, entcoeff, optim_epochs,
                                                                   optim_stepsize, optim_batchsize,
                                                                   gamma, lam, schedule,
                                                                   verbose, num_timesteps))
        super().__init__(env, output_path,
                         PPO1(policy=MlpPolicy,
                              env=env,
                              gamma=gamma,
                              timesteps_per_actorbatch=timesteps_per_actorbatch,
                              clip_param=clip_param,
                              entcoeff=entcoeff,
                              optim_epochs=optim_epochs,
                              optim_stepsize=optim_stepsize,
                              optim_batchsize=optim_batchsize,
                              lam=lam,
                              schedule=schedule,
                              verbose=verbose),
                         num_timesteps)


def create_env(env_id, output_path, seed=0):
    rank = MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed + 10000 * rank)
    env = gym.make(env_id)
    env = Monitor(env, os.path.join(output_path, str(rank)), allow_early_resets=True)
    env.seed(seed)
    return env
