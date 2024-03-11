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
"""Placeholder docstring"""
from __future__ import absolute_import

import json
import logging
import tempfile
from typing import Union, Optional, Dict

from six.moves.urllib.parse import urlparse

from sagemaker import image_uris, s3_utils
from sagemaker.amazon import validation
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.amazon.common import write_numpy_to_dense_tensor
from sagemaker.deprecations import renamed_warning
from sagemaker.estimator import EstimatorBase, _TrainingJob
from sagemaker.inputs import FileSystemInput, TrainingInput
from sagemaker.utils import sagemaker_timestamp, check_and_get_run_experiment_config
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.pipeline_context import runnable_by_pipeline
from sagemaker.workflow import is_pipeline_variable

logger = logging.getLogger(__name__)


class AmazonAlgorithmEstimatorBase(EstimatorBase):
    """Base class for Amazon first-party Estimator implementations.

    This class isn't intended to be instantiated directly.
    """

    feature_dim: hp = hp("feature_dim", validation.gt(0), data_type=int)
    mini_batch_size: hp = hp("mini_batch_size", validation.gt(0), data_type=int)
    repo_name: Optional[str] = None
    repo_version: Optional[str] = None

    DEFAULT_MINI_BATCH_SIZE: Optional[int] = None

    def __init__(
        self,
        role: Optional[Union[str, PipelineVariable]] = None,
        instance_count: Optional[Union[int, PipelineVariable]] = None,
        instance_type: Optional[Union[str, PipelineVariable]] = None,
        data_location: Optional[str] = None,
        enable_network_isolation: Union[bool, PipelineVariable] = False,
        **kwargs
    ):
        """Initialize an AmazonAlgorithmEstimatorBase.

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            instance_count (int or PipelineVariable): Number of Amazon EC2 instances to use
                for training. Required.
            instance_type (str or PipelineVariable): Type of EC2 instance to use for training,
                for example, 'ml.c4.xlarge'. Required.
            data_location (str or None): The s3 prefix to upload RecordSet
                objects to, expressed as an S3 url. For example
                "s3://example-bucket/some-key-prefix/". Objects will be saved in
                a unique sub-directory of the specified location. If None, a
                default data location will be used.
            enable_network_isolation (bool or PipelineVariable): Specifies whether container will
                run in network isolation mode. Network isolation mode restricts
                the container access to outside networks (such as the internet).
                Also known as internet-free mode (default: ``False``).
            **kwargs: Additional parameters passed to
                :class:`~sagemaker.estimator.EstimatorBase`.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        super(AmazonAlgorithmEstimatorBase, self).__init__(
            role,
            instance_count,
            instance_type,
            enable_network_isolation=enable_network_isolation,
            **kwargs
        )

        data_location = data_location or (
            s3_utils.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                "sagemaker-record-sets",
                with_end_slash=True,
            )
        )
        self._data_location = data_location

    def training_image_uri(self):
        """Placeholder docstring"""
        return image_uris.retrieve(
            self.repo_name,
            self.sagemaker_session.boto_region_name,
            version=self.repo_version,
        )

    def hyperparameters(self):
        """Placeholder docstring"""
        return hp.serialize_all(self)

    @property
    def data_location(self):
        """Placeholder docstring"""
        return self._data_location

    @data_location.setter
    def data_location(self, data_location: str):
        """Placeholder docstring"""
        if is_pipeline_variable(data_location):
            raise TypeError(
                "Invalid input: data_location should be a plain string "
                "rather than a pipeline variable - ({}).".format(type(data_location))
            )

        if not data_location.startswith("s3://"):
            raise ValueError(
                'Expecting an S3 URL beginning with "s3://". Got "{}"'.format(data_location)
            )
        if data_location[-1] != "/":
            data_location = data_location + "/"
        self._data_location = data_location

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor.

        Args:
            job_details: the returned job details from a describe_training_job
                API call.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded.

        Returns:
            dictionary: The transformed init_params
        """
        init_params = super(
            AmazonAlgorithmEstimatorBase, cls
        )._prepare_init_params_from_job_description(job_details, model_channel_name)

        # The hyperparam names may not be the same as the class attribute that holds them,
        # for instance: local_lloyd_init_method is called local_init_method. We need to map these
        # and pass the correct name to the constructor.
        for attribute, value in cls.__dict__.items():
            if isinstance(value, hp):
                if value.name in init_params["hyperparameters"]:
                    init_params[attribute] = init_params["hyperparameters"][value.name]

        del init_params["hyperparameters"]
        del init_params["image_uri"]
        return init_params

    def prepare_workflow_for_training(self, records=None, mini_batch_size=None, job_name=None):
        """Calls _prepare_for_training. Used when setting up a workflow.

        Args:
            records (:class:`~RecordSet`): The records to train this ``Estimator`` on.
            mini_batch_size (int or None): The size of each mini-batch to use when
                training. If ``None``, a default value will be used.
            job_name (str): Name of the training job to be created. If not
                specified, one is generated, using the base name given to the
                constructor if applicable.
        """
        self._prepare_for_training(
            records=records, mini_batch_size=mini_batch_size, job_name=job_name
        )

    def _prepare_for_training(self, records, mini_batch_size=None, job_name=None):
        """Set hyperparameters needed for training.

        Args:
            records (:class:`~RecordSet`): The records to train this ``Estimator`` on.
            mini_batch_size (int or None): The size of each mini-batch to use when
                training. If ``None``, a default value will be used.
            job_name (str): Name of the training job to be created. If not
                specified, one is generated, using the base name given to the
                constructor if applicable.
        """
        super(AmazonAlgorithmEstimatorBase, self)._prepare_for_training(job_name=job_name)

        feature_dim = None

        if isinstance(records, list):
            for record in records:
                if record.channel == "train":
                    feature_dim = record.feature_dim
                    break
            if feature_dim is None:
                raise ValueError("Must provide train channel.")
        else:
            feature_dim = records.feature_dim

        self.feature_dim = feature_dim
        self.mini_batch_size = mini_batch_size

    @runnable_by_pipeline
    def fit(
        self,
        records: "RecordSet",
        mini_batch_size: Optional[int] = None,
        wait: bool = True,
        logs: bool = True,
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
    ):
        """Fit this Estimator on serialized Record objects, stored in S3.

        ``records`` should be an instance of :class:`~RecordSet`. This
        defines a collection of S3 data files to train this ``Estimator`` on.

        Training data is expected to be encoded as dense or sparse vectors in
        the "values" feature on each Record. If the data is labeled, the label
        is expected to be encoded as a list of scalas in the "values" feature of
        the Record label.

        More information on the Amazon Record format is available at:
        https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html

        See :meth:`~AmazonAlgorithmEstimatorBase.record_set` to construct a
        ``RecordSet`` object from :class:`~numpy.ndarray` arrays.

        Args:
            records (:class:`~RecordSet`): The records to train this ``Estimator`` on
            mini_batch_size (int or None): The size of each mini-batch to use
                when training. If ``None``, a default value will be used.
            wait (bool): Whether the call should wait until the job completes
                (default: True).
            logs (bool): Whether to show the logs produced by the job. Only
                meaningful when wait is True (default: True).
            job_name (str): Training job name. If not specified, the estimator
                generates a default job name, based on the training image name
                and current timestamp.
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain four keys:
                'ExperimentName', 'TrialName', 'TrialComponentDisplayName' and 'RunName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
        """
        self._prepare_for_training(records, job_name=job_name, mini_batch_size=mini_batch_size)

        experiment_config = check_and_get_run_experiment_config(experiment_config)
        self.latest_training_job = _TrainingJob.start_new(
            self, records, experiment_config=experiment_config
        )
        if wait:
            self.latest_training_job.wait(logs=logs)

    def record_set(self, train, labels=None, channel="train", encrypt=False):
        """Build a :class:`~RecordSet` from a numpy :class:`~ndarray` matrix and label vector.

        For the 2D ``ndarray`` ``train``, each row is converted to a
        :class:`~Record` object. The vector is stored in the "values" entry of
        the ``features`` property of each Record. If ``labels`` is not None,
        each corresponding label is assigned to the "values" entry of the
        ``labels`` property of each Record.

        The collection of ``Record`` objects are protobuf serialized and
        uploaded to new S3 locations. A manifest file is generated containing
        the list of objects created and also stored in S3.

        The number of S3 objects created is controlled by the
        ``instance_count`` property on this Estimator. One S3 object is
        created per training instance.

        Args:
            train (numpy.ndarray): A 2D numpy array of training data.
            labels (numpy.ndarray): A 1D numpy array of labels. Its length must
                be equal to the number of rows in ``train``.
            channel (str): The SageMaker TrainingJob channel this RecordSet
                should be assigned to.
            encrypt (bool): Specifies whether the objects uploaded to S3 are
                encrypted on the server side using AES-256 (default: ``False``).

        Returns:
            RecordSet: A RecordSet referencing the encoded, uploading training
            and label data.
        """
        s3 = self.sagemaker_session.boto_session.resource(
            "s3", region_name=self.sagemaker_session.boto_region_name
        )
        parsed_s3_url = urlparse(self.data_location)
        bucket, key_prefix = parsed_s3_url.netloc, parsed_s3_url.path
        key_prefix = key_prefix + "{}-{}/".format(type(self).__name__, sagemaker_timestamp())
        key_prefix = key_prefix.lstrip("/")
        logger.debug("Uploading to bucket %s and key_prefix %s", bucket, key_prefix)
        manifest_s3_file = upload_numpy_to_s3_shards(
            self.instance_count, s3, bucket, key_prefix, train, labels, encrypt
        )
        logger.debug("Created manifest file %s", manifest_s3_file)
        return RecordSet(
            manifest_s3_file,
            num_records=train.shape[0],
            feature_dim=train.shape[1],
            channel=channel,
        )

    def _get_default_mini_batch_size(self, num_records: int):
        """Generate the default mini_batch_size"""
        if is_pipeline_variable(self.instance_count):
            logger.warning(
                "mini_batch_size is not given in .fit() and instance_count is a "
                "pipeline variable (%s) which is only interpreted in pipeline execution time. "
                "Thus setting mini_batch_size to 1, since it can't be greater than "
                "number of records per instance_count, otherwise the training job fails.",
                type(self.instance_count),
            )
            return 1

        return min(self.DEFAULT_MINI_BATCH_SIZE, max(1, int(num_records / self.instance_count)))


class RecordSet(object):
    """Placeholder docstring"""

    def __init__(
        self,
        s3_data: Union[str, PipelineVariable],
        num_records: int,
        feature_dim: int,
        s3_data_type: Union[str, PipelineVariable] = "ManifestFile",
        channel: Union[str, PipelineVariable] = "train",
    ):
        """A collection of Amazon :class:~`Record` objects serialized and stored in S3.

        Args:
            s3_data (str or PipelineVariable): The S3 location of the training data
            num_records (int): The number of records in the set.
            feature_dim (int): The dimensionality of "values" arrays in the
                Record features, and label (if each Record is labeled).
            s3_data_type (str or PipelineVariable): Valid values: 'S3Prefix', 'ManifestFile'.
                If 'S3Prefix', ``s3_data`` defines a prefix of s3 objects to train
                on. All objects with s3 keys beginning with ``s3_data`` will be
                used to train. If 'ManifestFile', then ``s3_data`` defines a
                single s3 manifest file, listing each s3 object to train on.
            channel (str or PipelineVariable): The SageMaker Training Job channel this RecordSet
                should be bound to
        """
        self.s3_data = s3_data
        self.feature_dim = feature_dim
        self.num_records = num_records
        self.s3_data_type = s3_data_type
        self.channel = channel

    def __repr__(self):
        """Return an unambiguous representation of this RecordSet"""
        return str((RecordSet, self.__dict__))

    def data_channel(self):
        """Returns dictionary to represent the training data in a channel to use with ``fit()``."""

        return {self.channel: self.records_s3_input()}

    def records_s3_input(self):
        """Return a TrainingInput to represent the training data"""
        return TrainingInput(
            self.s3_data, distribution="ShardedByS3Key", s3_data_type=self.s3_data_type
        )


class FileSystemRecordSet(object):
    """Amazon SageMaker channel configuration for file system data source for Amazon algorithms."""

    def __init__(
        self,
        file_system_id,
        file_system_type,
        directory_path,
        num_records,
        feature_dim,
        file_system_access_mode="ro",
        channel="train",
    ):
        """Initialize a ``FileSystemRecordSet`` object.

        Args:
            file_system_id (str): An Amazon file system ID starting with 'fs-'.
            file_system_type (str): The type of file system used for the input.
                Valid values: 'EFS', 'FSxLustre'.
            directory_path (str): Absolute or normalized path to the root directory (mount point) in
                the file system. Reference:
                https://docs.aws.amazon.com/efs/latest/ug/mounting-fs.html and
                https://docs.aws.amazon.com/efs/latest/ug/wt1-test.html
            num_records (int): The number of records in the set.
            feature_dim (int): The dimensionality of "values" arrays in the Record features,
                and label (if each Record is labeled).
            file_system_access_mode (str): Permissions for read and write.
                Valid values: 'ro' or 'rw'. Defaults to 'ro'.
            channel (str): The SageMaker Training Job channel this RecordSet should be bound to
        """

        self.file_system_input = FileSystemInput(
            file_system_id, file_system_type, directory_path, file_system_access_mode
        )
        self.feature_dim = feature_dim
        self.num_records = num_records
        self.channel = channel

    def __repr__(self):
        """Return an unambiguous representation of this RecordSet"""
        return str((FileSystemRecordSet, self.__dict__))

    def data_channel(self):
        """Return a dictionary to represent the training data in a channel for use with ``fit()``"""
        return {self.channel: self.file_system_input}


def _build_shards(num_shards, array):
    """Placeholder docstring"""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    shard_size = int(array.shape[0] / num_shards)
    if shard_size == 0:
        raise ValueError("Array length is less than num shards")
    shards = [array[i * shard_size : i * shard_size + shard_size] for i in range(num_shards - 1)]
    shards.append(array[(num_shards - 1) * shard_size :])
    return shards


def upload_numpy_to_s3_shards(
    num_shards, s3, bucket, key_prefix, array, labels=None, encrypt=False
):
    """Upload the training ``array`` and ``labels`` arrays to ``num_shards``.

    S3 objects, stored in "s3:// ``bucket`` / ``key_prefix`` /". Optionally
    ``encrypt`` the S3 objects using AES-256.

    Args:
        num_shards:
        s3:
        bucket:
        key_prefix:
        array:
        labels:
        encrypt:
    """
    shards = _build_shards(num_shards, array)
    if labels is not None:
        label_shards = _build_shards(num_shards, labels)
    uploaded_files = []
    if key_prefix[-1] != "/":
        key_prefix = key_prefix + "/"
    extra_put_kwargs = {"ServerSideEncryption": "AES256"} if encrypt else {}
    try:
        for shard_index, shard in enumerate(shards):
            with tempfile.TemporaryFile() as file:
                if labels is not None:
                    write_numpy_to_dense_tensor(file, shard, label_shards[shard_index])
                else:
                    write_numpy_to_dense_tensor(file, shard)
                file.seek(0)
                shard_index_string = str(shard_index).zfill(len(str(len(shards))))
                file_name = "matrix_{}.pbr".format(shard_index_string)
                key = key_prefix + file_name
                logger.debug("Creating object %s in bucket %s", key, bucket)
                s3.Object(bucket, key).put(Body=file, **extra_put_kwargs)
                uploaded_files.append(file_name)
        manifest_key = key_prefix + ".amazon.manifest"
        manifest_str = json.dumps(
            [{"prefix": "s3://{}/{}".format(bucket, key_prefix)}] + uploaded_files
        )
        s3.Object(bucket, manifest_key).put(Body=manifest_str.encode("utf-8"), **extra_put_kwargs)
        return "s3://{}/{}".format(bucket, manifest_key)
    except Exception as ex:  # pylint: disable=broad-except
        try:
            for file in uploaded_files:
                s3.Object(bucket, key_prefix + file).delete()
        finally:
            raise ex


def get_image_uri(region_name, repo_name, repo_version="1"):
    """Deprecated method. Please use sagemaker.image_uris.retrieve().

    Args:
        region_name: name of the region
        repo_name: name of the repo (e.g. xgboost)
        repo_version: version of the repo

    Returns:
        the image uri
    """
    renamed_warning("The method get_image_uri")
    return image_uris.retrieve(
        framework=repo_name,
        region=region_name,
        version=repo_version,
    )
