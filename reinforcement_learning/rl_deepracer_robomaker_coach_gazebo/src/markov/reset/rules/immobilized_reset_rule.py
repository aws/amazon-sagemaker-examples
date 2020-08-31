'''This module implements concrete reset rule for the immobilization'''

import markov.agent_ctrl.constants as const

from markov.reset.abstract_reset_rule import AbstractResetRule
from markov.reset.constants import AgentCtrlStatus, AgentPhase
from markov.metrics.constants import EpisodeStatus

class ImmobilizedResetRule(AbstractResetRule):
    name = EpisodeStatus.IMMOBILIZED.value

    def __init__(self):
        super(ImmobilizedResetRule, self).__init__(ImmobilizedResetRule.name)
        self.immobilize_count = 0

    def _update(self, agent_status):
        '''Update the immobilized reset rule done flag

        Args:
            agent_status (dict): agent status dictionary
        '''
        agent_phase = agent_status[AgentCtrlStatus.AGENT_PHASE.value]
        prev_pnt_dist = agent_status[AgentCtrlStatus.PREV_PNT_DIST.value]

        if agent_phase == AgentPhase.RUN.value and prev_pnt_dist <= 0.0001:
            self.immobilize_count += 1
        else:
            self.immobilize_count = 0
        if self.immobilize_count >= const.NUM_STEPS_TO_CHECK_STUCK:
            self._reverse_count = 0
            self._done = True

