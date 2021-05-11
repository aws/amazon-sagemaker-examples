import abc

import markov.gazebo_tracker.constants as consts
from markov.gazebo_tracker.tracker_manager import TrackerManager

# Python 2 and 3 compatible Abstract class
ABC = abc.ABCMeta("ABC", (object,), {})


class AbstractTracker(ABC):
    def __init__(self, priority=consts.TrackerPriority.NORMAL):
        self._priority = priority
        TrackerManager.get_instance().add(self, priority)

    @abc.abstractmethod
    def update_tracker(self, delta_time, sim_time):
        """
        Update Tracker

        Args:
            delta_time (float): delta time
            sim_time (Clock): simulation time
        """
        raise NotImplementedError("Tracker must be able to update")
