def reward_function(on_track, x, y, distance_from_center, car_orientation, progress, steps,
               throttle, steering, track_width, waypoints, closest_waypoints):

    if distance_from_center >= 0.0 and distance_from_center <= 0.4 * track_width:
        return 1.0
    else:
        return 0.1

    return 1e-3  # like crashed
