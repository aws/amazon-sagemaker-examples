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
def reward_function(on_track, x, y, distance_from_center, car_orientation, progress, steps,
               throttle, steering, track_width, waypoints, closest_waypoints):

    if distance_from_center >= 0.0 and distance_from_center <= 0.02:
        return 1.0
    elif distance_from_center >= 0.02 and distance_from_center <= 0.03:
        return 0.3
    elif distance_from_center >= 0.03 and distance_from_center <= 0.05:
        return 0.1

    return 1e-3  # like crashed

