'''This module should house utility methods for the track data classes'''
import bisect
import numpy as np
import math


# The order of rotation applied: roll -> pitch -> yaw
def euler_to_quaternion(roll=0, pitch=0, yaw=0):
    # Abbreviations for the various angular functions
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Quaternion
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    w = cy * cp * cr + sy * sp * sr

    return x, y, z, w


def quaternion_to_euler(x, y, z, w):
    '''convert quaternion x, y, z, w to euler angle roll, pitch, yaw

    Args:
        x: quaternion x
        y: quaternion y
        z: quaternion z
        w: quaternion w
    
    Returns:
        Tuple: (roll, pitch, yaw) tuple
    
    '''
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def inverse_quaternion(q, threshold=0.000001):
    q = np.array(q)
    n = np.dot(q, q)
    if n < threshold:
        raise Exception("Cannot calculate inverse with quaternion length is 0")
    inv_q = np.zeros(4)
    inv_q[0] = -q[0] / n
    inv_q[1] = -q[1] / n
    inv_q[2] = -q[2] / n
    inv_q[3] = q[3] / n
    return inv_q


def apply_orientation(q, v):
    # Make sure q and v are numpy.ndarray type
    q = np.array(q)
    v = np.array(v)

    u, s = q[:3], q[3]
    return 2.0 * np.dot(u, v) * u \
           + (s*s - np.dot(u, u)) * v \
           + 2.0 * s * np.cross(u, v)


def find_prev_next(a, x, reverse_dir):
    if reverse_dir:
        next_index = bisect.bisect_left(a, x) - 1
        prev_index = next_index + 1
        if next_index == -1: next_index = len(a) - 1
    else:
        next_index = bisect.bisect_right(a, x)
        prev_index = next_index - 1
        if next_index == len(a): next_index = 0
    return prev_index, next_index
