from gym.envs.registration import register

MAX_STEPS = 1000

register(
    id='SageMaker-DeepRacer-Discrete-v0',
    entry_point='robomaker.environments.deepracer_env:DeepRacerDiscreteEnv',
    max_episode_steps=MAX_STEPS,
    reward_threshold=200
)
