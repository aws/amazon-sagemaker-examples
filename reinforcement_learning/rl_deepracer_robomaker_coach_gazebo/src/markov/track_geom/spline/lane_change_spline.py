"""This module implements concrete lane change spline"""

import bisect
import random

import numpy as np
from markov.track_geom.constants import DIST_THRESHOLD, SPLINE_DEGREE
from markov.track_geom.spline.abstract_spline import AbstractSpline
from markov.track_geom.track_data import TrackLine
from scipy.interpolate import splprep
from shapely.geometry import Point
from shapely.geometry.polygon import LineString


class LaneChangeSpline(AbstractSpline):
    def __init__(
        self, start_lane, end_lane, current_dist, lane_change_start_dist, lane_change_end_dist
    ):
        self._start_lane = start_lane
        self._end_lane = end_lane
        self._current_dist = current_dist
        self._lane_change_start_dist = lane_change_start_dist
        self._lane_change_end_dist = lane_change_end_dist
        super(LaneChangeSpline, self).__init__()

    def _build_spline(self):
        """Build spline for lane change

        Returns:
            tuple: lane change lane, lane point distance,
                  prepared lane change spline.
        """
        # cetner line
        center_line = self._track_data.center_line

        # start lane
        start_lane_line = self._start_lane.lane["track_line"]
        start_lane_dists = self._start_lane.lane["dists"]

        # end lane
        end_lane_line = self._end_lane.lane["track_line"]
        end_lane_dists = self._end_lane.lane["dists"]

        start_lane_point = Point(
            np.array(self._start_lane.eval_spline(self._lane_change_start_dist))[:, 0]
        )
        end_lane_point = Point(
            np.array(self._end_lane.eval_spline(self._lane_change_end_dist))[:, 0]
        )
        end_offset = (
            0.0
            if (self._lane_change_start_dist < self._lane_change_end_dist)
            else center_line.length
        )

        # Find prev/next points on each lane
        current_prev_index = bisect.bisect_left(start_lane_dists, self._current_dist) - 1
        start_prev_index = bisect.bisect_left(start_lane_dists, self._lane_change_start_dist) - 1
        end_next_index = bisect.bisect_right(end_lane_dists, self._lane_change_end_dist)

        # Define intervals on start/end lanes to build the spline from
        num_start_coords = len(start_lane_line.coords)
        num_end_coords = len(end_lane_line.coords)
        if self._track_data.is_loop:
            num_start_coords -= 1
            num_end_coords -= 1
        start_index_0 = (current_prev_index - 3) % num_start_coords
        start_index_1 = start_prev_index
        end_index_0 = end_next_index
        end_index_1 = (end_next_index + 3) % num_end_coords

        # Grab waypoint indices for these intervals (some corner cases here...)
        if start_index_0 < start_index_1:
            start_indices = list(range(start_index_0, start_index_1 + 1))
            start_offsets = [0.0] * len(start_indices)
        else:
            start_indices_0 = list(range(start_index_0, num_start_coords))
            start_indices_1 = list(range(start_index_1 + 1))
            start_indices = start_indices_0 + start_indices_1
            start_offsets = [-center_line.length] * len(start_indices_0) + [0.0] * len(
                start_indices_1
            )
        if end_index_0 < end_index_1:
            end_indices = list(range(end_index_0, end_index_1 + 1))
            end_offsets = [end_offset] * len(end_indices)
        else:
            end_indices_0 = list(range(end_index_0, num_end_coords))
            end_indices_1 = list(range(end_index_1 + 1))
            end_indices = end_indices_0 + end_indices_1
            end_offsets = [end_offset] * len(end_indices_0) + [
                end_offset + center_line.length
            ] * len(end_indices_1)

        # Logic to avoid start and end point are too close to track waypoints
        before_start_lane_point = Point(np.array(start_lane_line.coords.xy)[:, start_indices[-1]])
        after_end_lane_point = Point(np.array(end_lane_line.coords.xy)[:, end_indices[0]])
        if before_start_lane_point.distance(start_lane_point) < DIST_THRESHOLD:
            # pop last index of start_indices
            start_indices.pop()
            start_offsets.pop()
        if after_end_lane_point.distance(end_lane_point) < DIST_THRESHOLD:
            # pop first index of end_indices
            end_indices.pop(0)
            end_offsets.pop(0)

        # Build the spline
        u = np.hstack(
            (
                np.array(start_lane_dists)[start_indices] + np.array(start_offsets),
                self._lane_change_start_dist,
                self._lane_change_end_dist + end_offset,
                np.array(end_lane_dists)[end_indices] + np.array(end_offsets),
            )
        )
        x = np.hstack(
            (
                np.array(start_lane_line.coords.xy)[:, start_indices],
                start_lane_point.xy,
                end_lane_point.xy,
                np.array(end_lane_line.coords.xy)[:, end_indices],
            )
        )
        u, ui = np.unique(u, return_index=True)
        x = x[:, ui]
        bot_car_spline, _ = splprep(x, u=u, k=SPLINE_DEGREE, s=0)

        return TrackLine(LineString(np.array(np.transpose(x)))), u, bot_car_spline
