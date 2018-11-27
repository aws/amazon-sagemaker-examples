import json

class TrainingSpecification:

    template = """    
{
    "TrainingSpecification": {
    "TrainingImage": "IMAGE_REPLACE_ME",
    "SupportedHyperParameters": [
        {
            "Description": "Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes",
            "Name": "max_leaf_nodes",
            "Type": "Integer",
            "Range": {
                "IntegerParameterRangeSpecification": {
                    "MinValue": "1",
                    "MaxValue": "100000"
                }
            },
            "IsTunable": true,
            "IsRequired": false,
            "DefaultValue": "100"
        }
    ],
    "SupportedTrainingInstanceTypes": INSTANCES_REPLACE_ME,
    "SupportsDistributedTraining": false,
    "MetricDefinitions": METRICS_REPLACE_ME,
    "TrainingChannels": CHANNELS_REPLACE_ME,
    "SupportedTuningJobObjectiveMetrics": TUNING_OBJECTIVES_REPLACE_ME
    }
}
"""

    def get_training_specification_dict(self, ecr_image, supports_gpu, supported_channels=None, supported_metrics=None, supported_tuning_job_objective_metrics=None):
        return json.loads(self.get_training_specification_json(ecr_image, supports_gpu, supported_channels, supported_metrics, supported_tuning_job_objective_metrics))

    def get_training_specification_json(self, ecr_image, supports_gpu, supported_channels=None, supported_metrics=None, supported_tuning_job_objective_metrics=None):
        if supported_channels is None:
            print("Please provide at least one supported channel")
            raise ValueError("Please provide at least one supported channel")

        if supported_metrics is None:
            supported_metrics = []
        if supported_tuning_job_objective_metrics is None:
            supported_tuning_job_objective_metrics = []

        return self.template.replace("IMAGE_REPLACE_ME", ecr_image) \
            .replace("INSTANCES_REPLACE_ME", self.get_supported_instances(supports_gpu)) \
            .replace("CHANNELS_REPLACE_ME", json.dumps([ob.__dict__ for ob in supported_channels], indent=4, sort_keys=True)) \
            .replace("METRICS_REPLACE_ME", json.dumps([ob.__dict__ for ob in supported_metrics], indent=4, sort_keys=True)) \
            .replace("TUNING_OBJECTIVES_REPLACE_ME", json.dumps([ob.__dict__ for ob in supported_tuning_job_objective_metrics], indent=4, sort_keys=True))

    @staticmethod
    def get_supported_instances(supports_gpu):
        cpu_list = ["ml.m4.xlarge","ml.m4.2xlarge","ml.m4.4xlarge","ml.m4.10xlarge","ml.m4.16xlarge","ml.m5.large","ml.m5.xlarge","ml.m5.2xlarge","ml.m5.4xlarge","ml.m5.12xlarge","ml.m5.24xlarge","ml.c4.xlarge","ml.c4.2xlarge","ml.c4.4xlarge","ml.c4.8xlarge","ml.c5.xlarge","ml.c5.2xlarge","ml.c5.4xlarge","ml.c5.9xlarge","ml.c5.18xlarge"]
        gpu_list = ["ml.p2.xlarge", "ml.p2.8xlarge", "ml.p2.16xlarge", "ml.p3.2xlarge", "ml.p3.8xlarge", "ml.p3.16xlarge"]

        list_to_return = cpu_list

        if supports_gpu:
            list_to_return = cpu_list + gpu_list

        return json.dumps(list_to_return)
