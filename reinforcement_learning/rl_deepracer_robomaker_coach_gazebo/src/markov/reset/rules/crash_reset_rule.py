'''This module implements concrete reset rule for crash'''

from markov.reset.abstract_reset_rule import AbstractResetRule
from markov.reset.constants import AgentCtrlStatus, AgentInfo
from markov.track_geom.track_data import TrackData
from markov.track_geom.constants import AgentPos
from markov.metrics.constants import EpisodeStatus

class CrashResetRule(AbstractResetRule):
    name = EpisodeStatus.CRASHED.value

    def __init__(self, agent_name):
        super(CrashResetRule, self).__init__(CrashResetRule.name) 
        self._track_data = TrackData.get_instance()
        self._agent_name = agent_name

    def _update(self, agent_status):
        '''Update the crash reset rule done flag

        Args:
            agent_status (dict): agent status dictionary

        Returns:
            dict: dictionary contains the agent crash info
        '''
        pos_dict = agent_status[AgentCtrlStatus.POS_DICT.value]
        is_crashed, crashed_object_name = self._track_data.is_racecar_collided(pos_dict[AgentPos.LINK_POINTS.value], \
                                                          self._agent_name)
        self._done = is_crashed
        return {AgentInfo.CRASHED_OBJECT_NAME.value: crashed_object_name}
