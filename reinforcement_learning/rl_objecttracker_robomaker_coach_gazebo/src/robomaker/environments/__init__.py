from gym.envs.registration import register

MAX_STEPS = 1000

register(
    id='SageMaker-TurtleBot3-Discrete-v0',
    entry_point='robomaker.environments.object_tracker_env:TurtleBot3ObjectTrackerAndFollowerDiscreteEnv',
    max_episode_steps = MAX_STEPS,
    reward_threshold = 200
)