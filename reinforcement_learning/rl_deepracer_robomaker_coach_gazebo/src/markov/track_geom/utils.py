"""This module should house utility methods for the track data classes"""
import bisect
import math

import numpy as np
from markov.track_geom.constants import HIDE_POS_DELTA, HIDE_POS_OFFSET, START_POS_OFFSET


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
    """convert quaternion x, y, z, w to euler angle roll, pitch, yaw

    Args:
        x: quaternion x
        y: quaternion y
        z: quaternion z
        w: quaternion w

    Returns:
        Tuple: (roll, pitch, yaw) tuple

    """
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
    """This function is used to rotate a vector in the oriention of the given quternion.

    This function assumes that v is a homogeneous quternion. That is the real part is zero.
    The complete explanation can be found in the link
    https://math.stackexchange.com/questions/40164/how-do-you-rotate-a-vector-by-a-unit-quaternion
    https://en.wikipedia.org/wiki/Quaternion#Hamilton_product

    On an highlevel. We want the vector v in the direction of the quternion q. We know that
    q * q_conj = 1

    p = q * v * q_conj, where p is pure quternion, same length as v in the direction of q.

    The simplified formula in the executed code is derived from the below equations

    quaternion_mult(q,r)
        b1, c1, d1, a1 = q  # Here a1 and a2 are real numbers, b1, c1, d1 are imaginary i,j,k
        b2, c2, d2, a2 = r
        return [
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2,
            a1*a2 - b1*b2 - c1*c2 - d1*d2
        ]

    apply_orientation(q, v):
        r = np.insert(v, 3, 0)
        q_conj = [-1*q[0],-1*q[1],-1*q[2], q[3]]
        return quaternion_mult(quaternion_mult(q,r), q_conj)[:3]

    If the vector is not pure quternion. Then in the below simplified solution the real value returned will be
    a2*( a1_sq + b1_sq + c1_sq + d1_sq)

    Arguments:
        q (numpy.ndarray): A quternion numpy array of shape (4,)
        v (numpy.ndarray): A vector on which orientation has to be applied. A numpy array of shape (3,)
    """
    b1, c1, d1, a1 = q
    b2, c2, d2 = v[0], v[1], v[2]

    a1_sq = a1 ** 2
    b1_sq = b1 ** 2
    c1_sq = c1 ** 2
    d1_sq = d1 ** 2

    return np.array(
        [
            b2 * (-c1_sq - d1_sq + b1_sq + a1_sq)
            + 2 * (-(a1 * c2 * d1) + (b1 * c1 * c2) + (b1 * d1 * d2) + (a1 * c1 * d2)),
            c2 * (c1_sq - d1_sq + a1_sq - b1_sq)
            + 2 * ((a1 * b2 * d1) + (b1 * b2 * c1) + (c1 * d1 * d2) - (a1 * b1 * d2)),
            d2 * (-c1_sq + d1_sq + a1_sq - b1_sq)
            + 2 * ((a1 * b1 * c2) + (b1 * b2 * d1) - (a1 * b2 * c1) + (c1 * c2 * d1)),
        ]
    )


def find_prev_next(a, x):
    next_index = bisect.bisect_right(a, x)
    prev_index = next_index - 1
    if prev_index == -1:
        prev_index = len(a) - 1
    if next_index == len(a):
        next_index = 0
    return prev_index, next_index


def pose_distance(pose_a, pose_b):
    p_a = pose_a.position
    p_b = pose_b.position
    return math.sqrt((p_b.x - p_a.x) ** 2 + (p_b.y - p_a.y) ** 2 + (p_b.z - p_a.z) ** 2)


def get_start_positions(race_car_num):
    return [-START_POS_OFFSET * idx for idx in range(race_car_num)]


def get_hide_positions(race_car_num):
    """Generate hide positions for cars what will be outside of the race track environment.
       So that idle cars are not visible to customers.

    Args:
        race_car_num (int): The number of race cars in current environment.

    Returns:
        list: List of hiding positions.
    """
    # TODO: Maybe implement some logic to make sure the park postion is always outside of the race track
    return [
        (-(HIDE_POS_OFFSET + HIDE_POS_DELTA * idx), HIDE_POS_OFFSET) for idx in range(race_car_num)
    ]
