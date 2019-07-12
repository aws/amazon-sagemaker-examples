def reward_function(on_track, x, y, distance_from_center, car_orientation, progress, steps,
               throttle, steering, track_width, waypoints, closest_waypoints):

    reward = 1e-3
    if distance_from_center >= 0.0 and distance_from_center <= 0.4 * track_width:
        reward = 1.0

    # penalize reward if the car is steering way too much
    ABS_STEERING_THRESHOLD = 0.5
    if abs(steering) > ABS_STEERING_THRESHOLD:
        reward *= 0.5
    
    return 1e-3  # like crashed
