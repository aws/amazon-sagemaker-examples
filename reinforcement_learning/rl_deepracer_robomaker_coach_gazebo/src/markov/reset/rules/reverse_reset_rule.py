"""This module implements concrete reset rule for going reversed direction"""
import rospy
from markov.metrics.constants import EpisodeStatus
from markov.reset.abstract_reset_rule import AbstractResetRule
from markov.reset.constants import AgentCtrlStatus, AgentPhase


class ReverseResetRule(AbstractResetRule):
    name = EpisodeStatus.REVERSED.value

    def __init__(self):
        super(ReverseResetRule, self).__init__(ReverseResetRule.name)
        self._number_of_reverse_counts = int(rospy.get_param("NUMBER_OF_REVERSE_COUNTS", 15))
        self._reverse_count = 0

    def _update(self, agent_status):
        """Update the reset rule done flag

        Args:
            agent_status (dict): agent status dictionary
        """
        agent_phase = agent_status[AgentCtrlStatus.AGENT_PHASE.value]
        current_progress = agent_status[AgentCtrlStatus.CURRENT_PROGRESS.value]
        prev_progress = agent_status[AgentCtrlStatus.PREV_PROGRESS.value]

        if agent_phase == AgentPhase.RUN.value and current_progress < prev_progress:
            self._reverse_count += 1
        else:
            self._reverse_count = 0
        if self._reverse_count >= self._number_of_reverse_counts:
            self._reverse_count = 0
            self._done = True
