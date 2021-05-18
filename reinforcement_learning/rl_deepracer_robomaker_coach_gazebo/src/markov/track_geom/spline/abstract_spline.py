"""This class defines an abstract spline"""
import abc

from markov.track_geom.constants import SPLINE_DEGREE
from markov.track_geom.track_data import TrackData
from scipy.interpolate import spalde

# Python 2 and 3 compatible Abstract class
ABC = abc.ABCMeta("ABC", (object,), {})


class AbstractSpline(ABC):
    def __init__(self):
        self._track_data = TrackData.get_instance()
        self.build_spline()

    @property
    def lane(self):
        """Lane getter"""
        return self._lane

    def build_spline(self):
        """Build spline for track"""
        track_line, dists, spline = self._build_spline()
        self._lane = {"track_line": track_line, "dists": dists, "spline": spline}

    @abc.abstractmethod
    def _build_spline(self):
        """Build spline for track

        Returns:
            tuple: input track lane, track lane point distance,
                  prepared track lane spline.

        Raises:
            NotImplementedError: Build spline method is not implemented
        """
        raise NotImplementedError("Build spline method is not implemented")

    def eval_spline(self, dist):
        """Use spline to generate point

        Args:
            dist (float): lane change dist

        Returns:
            spalde: Evaluate all derivatives of a B-spline.
            https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.interpolate.spalde.html
        """
        center_line = self._track_data.center_line
        min_dist = self._lane["spline"][0][SPLINE_DEGREE]
        max_dist = self._lane["spline"][0][-SPLINE_DEGREE - 1]
        if dist < min_dist:
            dist += center_line.length
        if dist > max_dist:
            dist -= center_line.length
        return spalde(max(min_dist, min(dist, max_dist)), self._lane["spline"])
