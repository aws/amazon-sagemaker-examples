'''This module implements concrete reset rule for the immobilization'''

import markov.agent_ctrl.constants as const

from markov.reset.abstract_reset_rule import AbstractResetRule
from markov.reset.constants import AgentCtrlStatus
from markov.track_geom.track_data import TrackData
from markov.metrics.constants import EpisodeStatus

class ImmobilizedResetRule(AbstractResetRule):
    name = EpisodeStatus.IMMOBILIZED.value

    def __init__(self):
        super(ImmobilizedResetRule, self).__init__(ImmobilizedResetRule.name)

    def _update(self, agent_status):
        '''Update the immobilized reset rule done flag

        Args:
            agent_status (dict): agent status dictionary
        '''
        prev_pnt_dist = agent_status[AgentCtrlStatus.PREV_PNT_DIST.value]
        steps = agent_status[AgentCtrlStatus.STEPS.value]
        is_immobilized = (prev_pnt_dist <= 0.0001 and steps % const.NUM_STEPS_TO_CHECK_STUCK == 0)
        self._done = is_immobilized
