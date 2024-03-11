from ._utils import _get_rule_config


def vanishing_gradient():
    rule_config = _get_rule_config("VanishingGradient")
    return rule_config


def similar_across_runs():
    rule_config = _get_rule_config("SimilarAcrossRuns")
    return rule_config


def weight_update_ratio():
    rule_config = _get_rule_config("WeightUpdateRatio")
    return rule_config


def all_zero():
    rule_config = _get_rule_config("AllZero")
    return rule_config


def exploding_tensor():
    rule_config = _get_rule_config("ExplodingTensor")
    return rule_config


def unchanged_tensor():
    rule_config = _get_rule_config("UnchangedTensor")
    return rule_config


def loss_not_decreasing():
    rule_config = _get_rule_config("LossNotDecreasing")
    return rule_config


def check_input_images():
    rule_config = _get_rule_config("CheckInputImages")
    return rule_config


def dead_relu():
    rule_config = _get_rule_config("DeadRelu")
    return rule_config


def confusion():
    rule_config = _get_rule_config("Confusion")
    return rule_config


def tree_depth():
    rule_config = _get_rule_config("TreeDepth")
    return rule_config


def class_imbalance():
    rule_config = _get_rule_config("ClassImbalance")
    return rule_config


def overfit():
    rule_config = _get_rule_config("Overfit")
    return rule_config


def tensor_variance():
    rule_config = _get_rule_config("TensorVariance")
    return rule_config


def overtraining():
    rule_config = _get_rule_config("Overtraining")
    return rule_config


def poor_weight_initialization():
    rule_config = _get_rule_config("PoorWeightInitialization")
    return rule_config


def saturated_activation():
    rule_config = _get_rule_config("SaturatedActivation")
    return rule_config


def nlp_sequence_ratio():
    rule_config = _get_rule_config("NLPSequenceRatio")
    return rule_config


def stalled_training_rule():
    rule_config = _get_rule_config("StalledTrainingRule")
    return rule_config


def feature_importance_overweight():
    rule_config = _get_rule_config("FeatureImportanceOverweight")
    return rule_config


def create_xgboost_report():
    rule_config = _get_rule_config("CreateXgboostReport")
    return rule_config
