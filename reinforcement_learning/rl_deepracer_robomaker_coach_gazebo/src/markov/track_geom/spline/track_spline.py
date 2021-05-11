"""This module implements concrete track spline"""

import numpy as np
from markov.track_geom.constants import SPLINE_DEGREE, TrackLane
from markov.track_geom.spline.abstract_spline import AbstractSpline
from scipy.interpolate import splprep
from shapely.geometry import Point


class TrackSpline(AbstractSpline):
    def __init__(self, lane_name):
        self._lane_name = lane_name
        super(TrackSpline, self).__init__()

    def _build_spline(self):
        """Build spline for track

        Returns:
            tuple: input track lane, track lane point distance,
                  prepared track lane spline.
        """
        center_line = self._track_data.center_line
        if self._lane_name == TrackLane.INNER_LANE.value:
            lane = self._track_data.inner_lane
        elif self._lane_name == TrackLane.OUTER_LANE.value:
            lane = self._track_data.outer_lane
        else:
            lane = self._track_data.center_line
        lane_dists = [center_line.project(Point(c)) for c in lane.coords]
        # projecting inner/outer lane into center line cannot
        # guarantee monotonic increase along starting and ending position
        # if wrap around along start (more than half of track length),
        # subtract track length
        for i in range(len(lane_dists)):
            if lane_dists[i] < 0.5 * center_line.length:
                break
            lane_dists[i] -= center_line.length
        # if wrap around along finish (less than half of track length),
        # add track length
        for i in range(len(lane_dists) - 1, 0, -1):
            if lane_dists[i] > 0.5 * center_line.length:
                break
            lane_dists[i] += center_line.length
        u, ui = np.unique(lane_dists, return_index=True)
        x = np.array(lane.coords.xy)[:, ui]
        if u[0] > 0.0:
            p0 = lane.interpolate(lane.project(Point(center_line.coords[0])))
            u[0] = 0.0
            x[:, :1] = p0.xy
        if u[-1] < center_line.length:
            pN = lane.interpolate(lane.project(Point(center_line.coords[-1])))
            u[-1] = center_line.length
            x[:, -1:] = pN.xy
        if self._track_data.is_loop:
            x[:, -1] = x[:, 0]
            lane_spline, _ = splprep(x, u=u, k=SPLINE_DEGREE, s=0, per=1)
        else:
            lane_spline, _ = splprep(x, u=u, k=SPLINE_DEGREE, s=0)

        return lane, lane_dists, lane_spline
