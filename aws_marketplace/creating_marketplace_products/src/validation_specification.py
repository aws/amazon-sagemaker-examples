import json

class ValidationSpecification:
    template = """
{    
    "ValidationSpecification": {
        "ValidationRole": "ROLE_REPLACE_ME",
        "ValidationProfiles": [
            {
                "ProfileName": "ValidationProfile1",
                "TrainingJobDefinition": {
                    "TrainingInputMode": "File",
                    "HyperParameters": {},
                    "InputDataConfig": [
                        {
                            "ChannelName": "CHANNEL_NAME_REPLACE_ME",
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataType": "S3Prefix",
                                    "S3Uri": "TRAIN_S3_INPUT_REPLACE_ME",
                                    "S3DataDistributionType": "FullyReplicated"
                                }
                            },
                            "ContentType": "CONTENT_TYPE_REPLACE_ME",
                            "CompressionType": "None",
                            "RecordWrapperType": "None"
                        }
                     ],
                    "OutputDataConfig": {
                        "KmsKeyId": "",
                        "S3OutputPath": "VALIDATION_S3_OUTPUT_REPLACE_ME/training-output"
                    },
                    "ResourceConfig": {
                        "InstanceType": "INSTANCE_TYPE_REPLACE_ME",
                        "InstanceCount": 1,
                        "VolumeSizeInGB": 10,
                        "VolumeKmsKeyId": ""
                    },
                    "StoppingCondition": {
                        "MaxRuntimeInSeconds": 3600
                    }
                },
                "TransformJobDefinition": {
                    "MaxConcurrentTransforms": 1,
                    "MaxPayloadInMB": 6,
                    "TransformInput": {
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "BATCH_S3_INPUT_REPLACE_ME"
                            }
                        },
                        "ContentType": "CONTENT_TYPE_REPLACE_ME",
                        "CompressionType": "None",
                        "SplitType": "Line"
                    },
                    "TransformOutput": {
                        "S3OutputPath": "VALIDATION_S3_OUTPUT_REPLACE_ME/batch-transform-output",
                        "Accept": "CONTENT_TYPE_REPLACE_ME",
                        "AssembleWith": "Line",
                        "KmsKeyId": ""
                    },
                    "TransformResources": {
                        "InstanceType": "INSTANCE_TYPE_REPLACE_ME",
                        "InstanceCount": 1
                    }
                }
            }
        ],
        "ValidationOutputS3Prefix": "VALIDATION_S3_OUTPUT_REPLACE_ME/validation-output",
        "ValidateForMarketplace": true
    }
}    
"""

    def get_validation_specification_dict(self, validation_role, training_channel_name, training_input, batch_transform_input, content_type, instance_type, output_s3_location):
        return json.loads(self.get_validation_specification_json(validation_role, training_channel_name, training_input, batch_transform_input, content_type, instance_type, output_s3_location))

    def get_validation_specification_json(self, validation_role, training_channel_name, training_input, batch_transform_input, content_type, instance_type, output_s3_location):

        return self.template.replace("ROLE_REPLACE_ME", validation_role)\
            .replace("CHANNEL_NAME_REPLACE_ME", training_channel_name)\
            .replace("TRAIN_S3_INPUT_REPLACE_ME", training_input)\
            .replace("BATCH_S3_INPUT_REPLACE_ME", batch_transform_input)\
            .replace("CONTENT_TYPE_REPLACE_ME", content_type)\
            .replace("INSTANCE_TYPE_REPLACE_ME", instance_type)\
            .replace("VALIDATION_S3_OUTPUT_REPLACE_ME", output_s3_location)
