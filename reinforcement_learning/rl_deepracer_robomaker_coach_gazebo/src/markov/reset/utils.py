'''Utility methods for the reset’s module'''

from markov.agent_ctrl.constants import ConfigParams
from markov.reset.reset_rules_manager import ResetRulesManager
from markov.reset.rules.crash_reset_rule import CrashResetRule
from markov.reset.rules.immobilized_reset_rule import ImmobilizedResetRule
from markov.reset.rules.episode_complete_reset_rule import EpisodeCompleteResetRule
from markov.reset.rules.off_track_reset_rule import OffTrackResetRule
from markov.reset.rules.reverse_reset_rule import ReverseResetRule

def construct_reset_rules_manager(config_dict):
    '''construct the reset reset rule manager

    Args:
        config_dict (dict): configuration dictionary

    Returns:
        ResetRulesManager: reset rules manager class instance
    '''
    reset_rules_manager = ResetRulesManager()
    reset_rules_manager.add(EpisodeCompleteResetRule(config_dict[ConfigParams.IS_CONTINUOUS.value],
                                                     config_dict[ConfigParams.NUMBER_OF_TRIALS.value]))
    reset_rules_manager.add(ImmobilizedResetRule())
    reset_rules_manager.add(OffTrackResetRule())
    reset_rules_manager.add(CrashResetRule(config_dict[ConfigParams.AGENT_NAME.value]))
    reset_rules_manager.add(ReverseResetRule())
    return reset_rules_manager
