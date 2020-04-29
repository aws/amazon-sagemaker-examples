'''This module houses the constants for the reset package'''
from enum import Enum
    
class RaceType(Enum):
    '''Enum containing the keys for race type
    '''
    TIME_TRIAL = "TIME_TRIAL"
    OBJECT_AVOIDANCE = "OBJECT_AVOIDANCE"
    HEAD_TO_BOT = "HEAD_TO_BOT"
    HEAD_TO_MODEL = "HEAD_TO_MODEL"

class AgentPhase(Enum):
    '''Enum containing the keys for agent phase
    '''
    PAUSE = "pause"
    RUN = "run"

class AgentCtrlStatus(Enum):
    '''Enum containing the keys for agent control status
    '''
    POS_DICT = "pos_dict"
    STEPS = "steps"
    CURRENT_PROGRESS = "current_progress"
    PREV_PROGRESS = "prev_progress"
    PREV_PNT_DIST = "prev_pnt_dist"
    AGENT_PHASE = "agent_phase"

    @classmethod
    def validate_dict(cls, input_dict):
        '''Will raise an exception if input dict does not contain all the keys in the enum'''
        for key in cls:
            _ = input_dict[key.value]

class AgentInfo(Enum):
    '''Enum containing the keys for the agent info status
    '''
    LAP_COUNT = "lap_count"
    CURRENT_PROGRESS = "current_progress"
    CRASHED_OBJECT_NAME = "crashed_object_name"

    @classmethod
    def validate_dict(cls, input_dict):
        '''Will raise an exception if input dict does not contain all the keys in the enum'''
        for key in cls:
            _ = input_dict[key.value]
