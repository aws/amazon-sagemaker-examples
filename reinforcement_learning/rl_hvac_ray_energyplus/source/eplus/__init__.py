from gym.envs.registration import register

register(
    id='large-office-v0',
    entry_point='eplus.envs:LargeOfficeEnv',
)

register(
    id='data-center-v0',
    entry_point='eplus.envs:DataCenterEnv'
)
