from gym.envs.registration import register

MAX_STEPS = 1000

register(
    id='DeepRacerRacetrack-v0',
    entry_point='markov.environments.deepracer_racetrack_env:DeepRacerRacetrackEnv',
    max_episode_steps=MAX_STEPS,
    reward_threshold=200
)

register(
    id='DeepRacerRacetrackCustomActionSpaceEnv-v0',
    entry_point='markov.environments.deepracer_racetrack_env:DeepRacerRacetrackCustomActionSpaceEnv',
    max_episode_steps=MAX_STEPS,
    reward_threshold=200
)
