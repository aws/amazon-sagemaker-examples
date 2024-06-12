# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Amazon SageMaker channel configurations for S3 data sources and file system data sources"""
from __future__ import absolute_import, print_function

from typing import Union, Optional, List
import attr

from sagemaker.workflow.entities import PipelineVariable

FILE_SYSTEM_TYPES = ["FSxLustre", "EFS"]
FILE_SYSTEM_ACCESS_MODES = ["ro", "rw"]


class TrainingInput(object):
    """Amazon SageMaker channel configurations for S3 data sources.

    Attributes:
        config (dict[str, dict]): A SageMaker ``DataSource`` referencing
            a SageMaker ``S3DataSource``.
    """

    def __init__(
        self,
        s3_data: Union[str, PipelineVariable],
        distribution: Optional[Union[str, PipelineVariable]] = None,
        compression: Optional[Union[str, PipelineVariable]] = None,
        content_type: Optional[Union[str, PipelineVariable]] = None,
        record_wrapping: Optional[Union[str, PipelineVariable]] = None,
        s3_data_type: Union[str, PipelineVariable] = "S3Prefix",
        instance_groups: Optional[List[Union[str, PipelineVariable]]] = None,
        input_mode: Optional[Union[str, PipelineVariable]] = None,
        attribute_names: Optional[List[Union[str, PipelineVariable]]] = None,
        target_attribute_name: Optional[Union[str, PipelineVariable]] = None,
        shuffle_config: Optional["ShuffleConfig"] = None,
    ):
        r"""Create a definition for input data used by an SageMaker training job.

        See AWS documentation on the ``CreateTrainingJob`` API for more details
        on the parameters.

        Args:
            s3_data (str or PipelineVariable): Defines the location of S3 data to train on.
            distribution (str or PipelineVariable): Valid values: ``'FullyReplicated'``,
                ``'ShardedByS3Key'`` (default: ``'FullyReplicated'``).
            compression (str or PipelineVariable): Valid values: ``'Gzip'``, ``None``
                (default: None). This is used only in Pipe input mode.
            content_type (str or PipelineVariable): MIME type of the input data
                (default: None).
            record_wrapping (str or PipelineVariable): Valid values: 'RecordIO'
                (default: None).
            s3_data_type (str or PipelineVariable): Valid values: ``'S3Prefix'``,
                ``'ManifestFile'``, ``'AugmentedManifestFile'``.
                If ``'S3Prefix'``, ``s3_data`` defines a prefix of s3 objects to train on.
                All objects with s3 keys beginning with ``s3_data`` will be used to train.
                If ``'ManifestFile'`` or ``'AugmentedManifestFile'``,
                then ``s3_data`` defines a
                single S3 manifest file or augmented manifest file respectively,
                listing the S3 data to train on. Both the ManifestFile and
                AugmentedManifestFile formats are described at `S3DataSource
                <https://docs.aws.amazon.com/sagemaker/latest/dg/API_S3DataSource.html>`_
                in the `Amazon SageMaker API reference`.
            instance_groups (list[str] or list[PipelineVariable]): Optional. A list of
                instance group names in string format that you specified while configuring
                a heterogeneous cluster using the :class:`sagemaker.instance_group.InstanceGroup`.
                S3 data will be sent to all instance groups in the specified list.
                For instructions on how to use InstanceGroup objects
                to configure a heterogeneous cluster
                through the SageMaker generic and framework estimator classes, see
                `Train Using a Heterogeneous Cluster
                <https://docs.aws.amazon.com/sagemaker/latest/dg/train-heterogeneous-cluster.html>`_
                in the *Amazon SageMaker developer guide*.
                (default: None)
            input_mode (str or PipelineVariable): Optional override for this channel's input mode
                (default: None). By default, channels will use the input mode defined on
                ``sagemaker.estimator.EstimatorBase.input_mode``, but they will ignore
                that setting if this parameter is set.

                    * None - Amazon SageMaker will use the input mode specified in the ``Estimator``
                    * 'File' - Amazon SageMaker copies the training dataset from the S3 location to
                        a local directory.
                    * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via
                        a Unix-named pipe.
                    * 'FastFile' - Amazon SageMaker streams data from S3 on demand instead of
                        downloading the entire dataset before training begins.

            attribute_names (list[str] or list[PipelineVariable]): A list of one or more attribute
                names to use that are found in a specified AugmentedManifestFile.
            target_attribute_name (str or PipelineVariable): The name of the attribute will be
                predicted (classified) in a SageMaker AutoML job. It is required if the input is
                for SageMaker AutoML job.
            shuffle_config (sagemaker.inputs.ShuffleConfig): If specified this configuration enables
                shuffling on this channel. See the SageMaker API documentation for more info:
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_ShuffleConfig.html
        """
        self.config = {
            "DataSource": {"S3DataSource": {"S3DataType": s3_data_type, "S3Uri": s3_data}}
        }

        if not (target_attribute_name or distribution):
            distribution = "FullyReplicated"

        if distribution is not None:
            self.config["DataSource"]["S3DataSource"]["S3DataDistributionType"] = distribution

        if compression is not None:
            self.config["CompressionType"] = compression
        if content_type is not None:
            self.config["ContentType"] = content_type
        if record_wrapping is not None:
            self.config["RecordWrapperType"] = record_wrapping
        if instance_groups is not None:
            self.config["DataSource"]["S3DataSource"]["InstanceGroupNames"] = instance_groups
        if input_mode is not None:
            self.config["InputMode"] = input_mode
        if attribute_names is not None:
            self.config["DataSource"]["S3DataSource"]["AttributeNames"] = attribute_names
        if target_attribute_name is not None:
            self.config["TargetAttributeName"] = target_attribute_name
        if shuffle_config is not None:
            self.config["ShuffleConfig"] = {"Seed": shuffle_config.seed}


class ShuffleConfig(object):
    """For configuring channel shuffling using a seed.

    For more detail, see the AWS documentation:
    https://docs.aws.amazon.com/sagemaker/latest/dg/API_ShuffleConfig.html
    """

    def __init__(self, seed):
        """Create a ShuffleConfig.

        Args:
            seed (long): the long value used to seed the shuffled sequence.
        """
        self.seed = seed


@attr.s
class CreateModelInput(object):
    """A class containing parameters which can be used to create a SageMaker Model

    Parameters:
        instance_type (str): type or EC2 instance will be used for model deployment.
        accelerator_type (str): elastic inference accelerator type.
    """

    instance_type: str = attr.ib(default=None)
    accelerator_type: str = attr.ib(default=None)


@attr.s
class TransformInput(object):
    """Creates a class containing parameters for configuring input data for a batch tramsform job.

    It can be used when calling ``sagemaker.transformer.Transformer.transform()``

    Args:
        data (str): The S3 location of the input data that the model can consume.
        data_type (str): The data type for a batch transform job.
            (default: ``'S3Prefix'``)
        content_type (str): The multi-purpose internet email extension (MIME) type of the data.
            (default: None)
        compression_type (str): If your transform data is compressed, specify the compression type.
            Valid values: ``'Gzip'``, ``None``
            (default: None)
        split_type (str): The method to use to split the transform job's data files into smaller
            batches.
            Valid values: ``'Line'``, ``RecordIO``, ``'TFRecord'``, None
            (default: None)
        input_filter (str): A JSONPath expression for selecting a portion of the input data to pass
            to the algorithm. For example, you can use this parameter to exclude fields, such as an
            ID column, from the input. If you want SageMaker to pass the entire input dataset to the
            algorithm, accept the default value ``$``. For more information on batch transform data
            processing, input, join, and output, see
            `Associate Prediction Results with Input Records
            <https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html?>`_
            in the *Amazon SageMaker developer guide*.
            Example value: ``$``. For more information about valid values for this parameter, see
            `JSONPath Operators
            <https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html#data-processing-operators>`_
            in the *Amazon SageMaker developer guide*.
            (default: ``$``)
        output_filter (str): A JSONPath expression for selecting a portion of the joined dataset to
            save in the output file for a batch transform job. If you want SageMaker to store the
            entire input dataset in the output file, leave the default value, $. If you specify
            indexes that aren't within the dimension size of the joined dataset, you get an error.
            Example value: ``$``. For more information about valid values for this parameter, see
            `JSONPath Operators
            <https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html#data-processing-operators>`_
            in the *Amazon SageMaker developer guide*.
            (default: ``$``)
        join_source (str): Specifies the source of the data to join with the transformed data.
            The default value is ``None``, which specifies not to join the input with the
            transformed data. If you want the batch transform job to join the original input data
            with the transformed data, set to ``Input``.
            Valid values: ``None``, ``Input``
            (default: None)
        model_client_config (dict): Configures the timeout and maximum number of retries for
            processing a transform job invocation.

                * ``'InvocationsTimeoutInSeconds'`` (int) - The timeout value in seconds for an
                  invocation request. The default value is 600.
                * ``'InvocationsMaxRetries'`` (int) - The maximum number of retries when invocation
                  requests are failing.

            (default: ``{600,3}``)
        batch_data_capture_config (dict): The dict is an object of `BatchDataCaptureConfig
            <https://sagemaker.readthedocs.io/en/stable/api/utility/inputs.html#sagemaker.inputs.BatchDataCaptureConfig>`_
            and specifies configuration related to batch transform job
            for use with Amazon SageMaker Model Monitoring. For more information,
            see `Capture data from batch transform job
            <https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-capture-batch.html>`_
            in the *Amazon SageMaker developer guide*.
            (default: None)
    """

    data: str = attr.ib()
    data_type: str = attr.ib(default="S3Prefix")
    content_type: str = attr.ib(default=None)
    compression_type: str = attr.ib(default=None)
    split_type: str = attr.ib(default=None)
    input_filter: str = attr.ib(default=None)
    output_filter: str = attr.ib(default=None)
    join_source: str = attr.ib(default=None)
    model_client_config: dict = attr.ib(default=None)
    batch_data_capture_config: dict = attr.ib(default=None)


class FileSystemInput(object):
    """Amazon SageMaker channel configurations for file system data sources.

    Attributes:
        config (dict[str, dict]): A Sagemaker File System ``DataSource``.
    """

    def __init__(
        self,
        file_system_id,
        file_system_type,
        directory_path,
        file_system_access_mode="ro",
        content_type=None,
    ):
        """Create a new file system input used by an SageMaker training job.

        Args:
            file_system_id (str): An Amazon file system ID starting with 'fs-'.
            file_system_type (str): The type of file system used for the input.
                Valid values: 'EFS', 'FSxLustre'.
            directory_path (str): Absolute or normalized path to the root directory (mount point) in
                the file system.
                Reference: https://docs.aws.amazon.com/efs/latest/ug/mounting-fs.html and
                https://docs.aws.amazon.com/fsx/latest/LustreGuide/mount-fs-auto-mount-onreboot.html
            file_system_access_mode (str): Permissions for read and write.
                Valid values: 'ro' or 'rw'. Defaults to 'ro'.
        """

        if file_system_type not in FILE_SYSTEM_TYPES:
            raise ValueError(
                "Unrecognized file system type: %s. Valid values: %s."
                % (file_system_type, ", ".join(FILE_SYSTEM_TYPES))
            )

        if file_system_access_mode not in FILE_SYSTEM_ACCESS_MODES:
            raise ValueError(
                "Unrecognized file system access mode: %s. Valid values: %s."
                % (file_system_access_mode, ", ".join(FILE_SYSTEM_ACCESS_MODES))
            )

        self.config = {
            "DataSource": {
                "FileSystemDataSource": {
                    "FileSystemId": file_system_id,
                    "FileSystemType": file_system_type,
                    "DirectoryPath": directory_path,
                    "FileSystemAccessMode": file_system_access_mode,
                }
            }
        }

        if content_type:
            self.config["ContentType"] = content_type


class BatchDataCaptureConfig(object):
    """Configuration object passed in when create a batch transform job.

    Specifies configuration related to batch transform job data capture for use with
    Amazon SageMaker Model Monitoring
    """

    def __init__(
        self,
        destination_s3_uri: str,
        kms_key_id: str = None,
        generate_inference_id: bool = None,
    ):
        """Create new BatchDataCaptureConfig

        Args:
            destination_s3_uri (str): S3 Location to store the captured data
            kms_key_id (str): The KMS key to use when writing to S3.
                KmsKeyId can be an ID of a KMS key, ARN of a KMS key, alias of a KMS key,
                or alias of a KMS key. The KmsKeyId is applied to all outputs.
                (default: None)
            generate_inference_id (bool): Flag to generate an inference id
                (default: None)
        """
        self.destination_s3_uri = destination_s3_uri
        self.kms_key_id = kms_key_id
        self.generate_inference_id = generate_inference_id

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        batch_data_capture_config = {
            "DestinationS3Uri": self.destination_s3_uri,
        }

        if self.kms_key_id is not None:
            batch_data_capture_config["KmsKeyId"] = self.kms_key_id
        if self.generate_inference_id is not None:
            batch_data_capture_config["GenerateInferenceId"] = self.generate_inference_id

        return batch_data_capture_config
