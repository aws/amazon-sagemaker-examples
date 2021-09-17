"""This module implements concrete reset rule for the immobilization"""

import markov.agent_ctrl.constants as const
from markov.metrics.constants import EpisodeStatus
from markov.reset.abstract_reset_rule import AbstractResetRule
from markov.reset.constants import AgentCtrlStatus, AgentPhase


class ImmobilizedResetRule(AbstractResetRule):
    name = EpisodeStatus.IMMOBILIZED.value

    def __init__(self):
        super(ImmobilizedResetRule, self).__init__(ImmobilizedResetRule.name)
        self._immobilize_count = 0

    def _update(self, agent_status):
        """Update the immobilized reset rule done flag

        Args:
            agent_status (dict): agent status dictionary
        """
        agent_phase = agent_status[AgentCtrlStatus.AGENT_PHASE.value]
        current_progress = agent_status[AgentCtrlStatus.CURRENT_PROGRESS.value]
        prev_progress = agent_status[AgentCtrlStatus.PREV_PROGRESS.value]

        if agent_phase == AgentPhase.RUN.value and abs(current_progress - prev_progress) <= 0.0001:
            self._immobilize_count += 1
        else:
            self._immobilize_count = 0
        if self._immobilize_count >= const.NUM_STEPS_TO_CHECK_STUCK:
            self._immobilize_count = 0
            self._done = True
