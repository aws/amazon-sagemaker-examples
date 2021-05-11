import numpy as np
from gym.spaces import Box, Discrete


def box_space_from_description(dimension_list):
    """Takes a list of dimension descriptions, and returns a gym.space.Box.
    Each dimension is described by a tuple of (name, low_value, high_value, description).
    Currently name & description are unused.
    """
    lows = []
    highs = []
    for d in dimension_list:
        lows.append(d[1])
        highs.append(d[2])
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    shape = None  # Because it gets shape from lows & highs
    return Box(lows, highs, shape, dtype=np.float32)
