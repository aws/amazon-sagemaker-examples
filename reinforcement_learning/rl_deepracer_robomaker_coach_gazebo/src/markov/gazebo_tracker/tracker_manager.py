import logging
import threading

import markov.gazebo_tracker.constants as consts
import rospy
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.deepracer_exceptions import GenericRolloutException
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger
from rosgraph_msgs.msg import Clock

logger = Logger(__name__, logging.INFO).get_logger()


class TrackerManager(object):
    """
    TrackerManager class
    """

    _instance_ = None

    @staticmethod
    def get_instance():
        """Method for getting a reference to the Tracker Manager object"""
        if TrackerManager._instance_ is None:
            TrackerManager()
        return TrackerManager._instance_

    def __init__(self):
        if TrackerManager._instance_ is not None:
            raise GenericRolloutException("Attempting to construct multiple TrackerManager")
        self.priority_order = [
            consts.TrackerPriority.HIGH,
            consts.TrackerPriority.NORMAL,
            consts.TrackerPriority.LOW,
        ]
        self.tracker_map = {}
        for priority in self.priority_order:
            self.tracker_map[priority] = set()
        self.lock = threading.RLock()
        self.last_time = 0.0
        rospy.Subscriber("/clock", Clock, self._update_sim_time)
        TrackerManager._instance_ = self

    def add(self, tracker, priority=consts.TrackerPriority.NORMAL):
        """
        Add given tracker to manager

        Args:
            tracker (AbstractTracker): tracker object
            priority (TrackerPriority): prioirity
        """
        with self.lock:
            self.tracker_map[priority].add(tracker)

    def remove(self, tracker):
        """
        Remove given tracker from manager

        Args:
            tracker (AbstractTracker): tracker
        """
        with self.lock:
            for priority in self.priority_order:
                self.tracker_map[priority].discard(tracker)

    def _update_sim_time(self, sim_time):
        """
        Callback when sim time is updated

        Args:
            sim_time (Clock): simulation time
        """
        curr_time = sim_time.clock.secs + 1.0e-9 * sim_time.clock.nsecs
        if self.last_time is None:
            self.last_time = curr_time
        delta_time = curr_time - self.last_time
        lock_acquired = self.lock.acquire(False)
        if lock_acquired:
            try:
                self.last_time = curr_time
                for priority in self.priority_order:
                    copy_trackers = self.tracker_map[priority].copy()
                    for tracker in copy_trackers:
                        tracker.update_tracker(delta_time, sim_time)
            except Exception as e:
                log_and_exit(
                    "Tracker raised Exception: {}".format(e),
                    SIMAPP_SIMULATION_WORKER_EXCEPTION,
                    SIMAPP_EVENT_ERROR_CODE_500,
                )
            finally:
                self.lock.release()
        else:
            logger.info("TrackerManager: missed an _update_sim_time call")
