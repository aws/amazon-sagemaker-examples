import math
import numpy as np


def get_angle_between_two_points_2d_rad(pt1, pt2):
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        if dx == 0:
            return math.pi
        else:
            return math.atan2(dy, dx)


def lerp(current, target, fraction):
    fraction = np.clip(fraction, 0.0, 1.0)
    return current + fraction * (target - current)


def lerp_angle_rad(current, target, fraction):
    while current > target + np.pi:
        target += np.pi * 2.0
    while target > current + np.pi:
        target -= np.pi * 2.0
    return lerp(current, target, fraction)


def normalize(vector):
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
