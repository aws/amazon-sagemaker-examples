'''This module implement reset rule manager'''
from markov.deepracer_exceptions import GenericRolloutException
from markov.reset.abstract_reset_rule import AbstractResetRule
from markov.reset.constants import AgentCtrlStatus, AgentInfo
from markov.metrics.constants import EpisodeStatus

class ResetRulesManager():
    def __init__(self):
        self._reset_rules = dict()

    def get_dones(self):
        '''Get a dictionary of reset rules done flag
        '''
        rules_dict = {reset_type.value: self._reset_rules[reset_type.value].done \
                      if reset_type.value in self._reset_rules else False \
                      for reset_type in EpisodeStatus}
        return rules_dict

    def add(self, reset_rule):
        '''Add a reset rule into manager

        Args:
            reset_rule (AbstractResetRule): composite reset rule class instance

        Raises:
            GenericRolloutException: ResetRule object passed is not an object of AbstractResetRule
        '''
        if not isinstance(reset_rule, AbstractResetRule):
            raise GenericRolloutException("ResetRule object passed is not an object of AbstractResetRule")
        self._reset_rules[reset_rule.name] = reset_rule

    def reset(self):
        '''Reset reset rules done flag to False
        '''
        for reset_rule in self._reset_rules.values():
            reset_rule.reset()

    def update(self, agent_status):
        '''Update reset rules done flag based on the agent status

        Args:
            agent_status (dict): dictionary contains the agent status

        Returns:
            dict: dictionary contains the agent info after desired action is taken

        Raises:
            GenericRolloutException: key is not found in AgentCtrlStatus enum
        '''
        try:
            AgentCtrlStatus.validate_dict(agent_status)
        except KeyError as ex:
            raise GenericRolloutException("Key {}, not found".format(ex))
        agent_info_map = {}
        for reset_rule in self._reset_rules.values():
            agent_info = reset_rule.update(agent_status)
            agent_info_map.update(agent_info)
        try:
            AgentInfo.validate_dict(agent_info_map)
        except KeyError as ex:
            raise GenericRolloutException("Key {}, not found".format(ex))
        return agent_info_map
