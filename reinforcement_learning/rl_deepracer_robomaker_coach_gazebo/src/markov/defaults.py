# Default max number of steps to allow per episode
MAX_STEPS = 10000
# Default enviroment
ENV_ID = 'DeepRacerRacetrackCustomActionSpaceEnv-v0'
# Entry point for default enviroment
ENTRY_POINT = 'markov.environments.deepracer_racetrack_env:DeepRacerRacetrackCustomActionSpaceEnv'
# Default reward threshold
THRESHOLD = 200

"""
Default action space from re:Invent (6 actions).
"""
model_metadata = {
    "action_space": [
        {
            "steering_angle": 45,
            "speed": 0.8
        },
        {
            "steering_angle": -45,
            "speed": 0.8
        },
        {
            "steering_angle": 0,
            "speed": 0.8
        },
        {
            "steering_angle": 22.5,
            "speed": 0.8
        },
        {
            "steering_angle": -22.5,
            "speed": 0.8
        },
        {
            "steering_angle": 0,
            "speed": 0.4
        }
    ]
}

"""
Default reward function is the centerline.
"""
def reward_function(params):

    distance_from_center = params['distance_from_center']
    track_width = params['track_width']

    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    reward = 1e-3
    if distance_from_center <= marker_1:
        reward = 1
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track

    return float(reward)
