def reward_function(on_track, x, y, distance_from_center, car_orientation, progress, steps,
               throttle, steering, track_width, waypoints, closest_waypoints):

    if distance_from_center >= 0.0 and distance_from_center <= 0.02:
        return 1.0
    elif distance_from_center >= 0.02 and distance_from_center <= 0.03:
        return 0.3
    elif distance_from_center >= 0.03 and distance_from_center <= 0.05:
        return 0.1

    return 1e-3  # like crashed
