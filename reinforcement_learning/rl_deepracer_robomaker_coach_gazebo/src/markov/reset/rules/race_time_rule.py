"""This module implements concrete reset rule for race time up"""
import logging

from markov.log_handler.logger import Logger
from markov.metrics.constants import EpisodeStatus
from markov.reset.abstract_reset_rule import AbstractResetRule
from markov.reset.constants import RaceCtrlStatus
from markov.track_geom.track_data import TrackData

LOG = Logger(__name__, logging.INFO).get_logger()


class RaceTimeRule(AbstractResetRule):
    name = EpisodeStatus.TIME_UP.value

    def __init__(self, race_duration):
        super(RaceTimeRule, self).__init__(RaceTimeRule.name)
        self._race_duration = race_duration

    def _update(self, agent_status):
        """Update the race time up flag

        Args:
            agent_status (dict): agent status dictionary
        """
        start_time = agent_status[RaceCtrlStatus.RACE_START_TIME.value]
        current_time = agent_status[RaceCtrlStatus.RACE_CURR_TIME.value]
        if (current_time - start_time) > self._race_duration:
            LOG.info("issue done. start_time: %s, current_time: %s", start_time, current_time)
            self._done = True
