from enum import Enum

from ._utils import _get_config_for_group


class MXNET(Enum):
    SIMPLE = ["VanishingGradient", "LossNotDecreasing", "WeightUpdateRatio"]
    ALL = []


class TENSORFLOW(Enum):
    SIMPLE = ["VanishingGradient", "LossNotDecreasing", "WeightUpdateRatio"]
    ALL = []


class PYTORCH(Enum):
    SIMPLE = ["VanishingGradient", "LossNotDecreasing", "WeightUpdateRatio"]
    ALL = []


class XGBOOST(Enum):
    SIMPLE = ["TreeDepth", "ClassImbalance"]
    ALL = []


def get_rule_groups(ruleEnum):
    ruleEnumVal = ruleEnum.value
    rules_config = _get_config_for_group(ruleEnumVal)
    return rules_config
