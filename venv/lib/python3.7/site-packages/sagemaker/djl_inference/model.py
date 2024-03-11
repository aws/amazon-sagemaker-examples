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
import os.path
import urllib.request
from json import JSONDecodeError
from urllib.error import HTTPError, URLError
from enum import Enum
from typing import Optional, Union, Dict, Any, List

import sagemaker
from sagemaker import s3, Predictor, image_uris, fw_utils
from sagemaker.deserializers import JSONDeserializer, BaseDeserializer
from sagemaker.djl_inference import defaults
from sagemaker.model import FrameworkModel
from sagemaker.s3_utils import s3_path_join
from sagemaker.serializers import JSONSerializer, BaseSerializer
from sagemaker.session import Session
from sagemaker.utils import _tmpdir, _create_or_update_code_dir
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.estimator import Estimator
from sagemaker.s3 import S3Uploader

logger = logging.getLogger("sagemaker")

# DJL Serving uses log4j, so we convert python logging level to log4j equivalent
_LOG_LEVEL_MAP = {
    logging.INFO: "info",
    logging.DEBUG: "debug",
    logging.WARNING: "warn",
    logging.ERROR: "error",
    logging.FATAL: "fatal",
    logging.CRITICAL: "fatal",
    logging.NOTSET: "off",
}


class DJLServingEngineEntryPointDefaults(Enum):
    """Enum describing supported engines and corresponding default inference handler modules."""

    DEEPSPEED = ("DeepSpeed", "djl_python.deepspeed")
    HUGGINGFACE_ACCELERATE = ("Python", "djl_python.huggingface")
    STABLE_DIFFUSION = ("DeepSpeed", "djl_python.stable-diffusion")
    FASTER_TRANSFORMER = ("FasterTransformer", "djl_python.fastertransformer")


class DJLPredictor(Predictor):
    """A Predictor for inference against DJL Model Endpoints.

    This is able to serialize Python lists, dictionaries, and numpy arrays to
    multidimensional tensors for DJL inference.
    """

    def __init__(
        self,
        endpoint_name: str,
        sagemaker_session: Session = None,
        serializer: BaseSerializer = JSONSerializer(),
        deserializer: BaseDeserializer = JSONDeserializer(),
    ):
        """Initialize a ``DJLPredictor``

        Args:
            endpoint_name (str): The name of the endpoint to perform inference
                on.
            sagemaker_session (sagemaker.session.Session): Session object that
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            serializer (sagemaker.serializers.BaseSerializer): Optional. Default
                serializes input data to json format.
            deserializer (sagemaker.deserializers.BaseDeserializer): Optional.
                Default parses the response from json format to dictionary.
        """
        super(DJLPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )


def _determine_engine_for_model(model_type: str, num_partitions: int, num_heads: int):
    """Placeholder docstring"""

    # Tensor Parallelism is only possible if attention heads can be split evenly
    # across devices
    if num_heads is not None and num_partitions is not None and num_heads % num_partitions:
        return HuggingFaceAccelerateModel
    if model_type in defaults.DEEPSPEED_RECOMMENDED_ARCHITECTURES:
        return DeepSpeedModel
    if model_type in defaults.FASTER_TRANSFORMER_RECOMMENDED_ARCHITECTURES:
        return FasterTransformerModel
    return HuggingFaceAccelerateModel


def _validate_engine_for_model_type(cls, model_type: str, num_partitions: int, num_heads: int):
    """Placeholder docstring"""

    if cls == DeepSpeedModel:
        if num_heads is not None and num_partitions is not None and num_heads % num_partitions:
            raise ValueError(
                "The number of attention heads is not evenly divisible by the number of partitions."
                "Please set the number of partitions such that the number of attention heads can be"
                "evenly split across the partitions."
            )
    if cls == FasterTransformerModel:
        if model_type not in defaults.FASTER_TRANSFORMER_SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"The model architecture {model_type} is currently not supported by "
                f"FasterTransformer. Please use a different engine, or use the DJLModel"
                f"to let SageMaker pick a recommended engine for this model."
            )
    return cls


def _read_existing_serving_properties(directory: str):
    """Placeholder docstring"""

    serving_properties_path = os.path.join(directory, "serving.properties")
    properties = {}
    if os.path.exists(serving_properties_path):
        with open(serving_properties_path, "r") as f:
            for line in f:
                if line.startswith("#") or len(line.strip()) == 0:
                    continue
                key, val = line.split("=", 1)
                properties[key] = val
    return properties


def _get_model_config_properties_from_s3(model_s3_uri: str, sagemaker_session: Session):
    """Placeholder docstring"""

    s3_files = s3.S3Downloader.list(model_s3_uri, sagemaker_session=sagemaker_session)
    model_config = None
    for config in defaults.VALID_MODEL_CONFIG_FILES:
        config_file = os.path.join(model_s3_uri, config)
        if config_file in s3_files:
            model_config = json.loads(
                s3.S3Downloader.read_file(config_file, sagemaker_session=sagemaker_session)
            )
            break
    if not model_config:
        raise ValueError(
            f"Did not find a config.json or model_index.json file in {model_s3_uri}. Please make "
            f"sure a config.json exists (or model_index.json for Stable Diffusion Models) in"
            f"the provided s3 location"
        )
    return model_config


def _get_model_config_properties_from_hf(model_id: str):
    """Placeholder docstring"""

    config_url_prefix = f"https://huggingface.co/{model_id}/raw/main/"
    model_config = None
    for config in defaults.VALID_MODEL_CONFIG_FILES:
        config_file_url = config_url_prefix + config
        try:
            with urllib.request.urlopen(config_file_url) as response:
                model_config = json.load(response)
                break
        except (HTTPError, URLError, TimeoutError, JSONDecodeError) as e:
            logger.warning(
                "Exception encountered while trying to read config file %s. " "Details: %s",
                config_file_url,
                e,
            )
    if not model_config:
        raise ValueError(
            f"Did not find a config.json or model_index.json file in huggingface hub for "
            f"{model_id}. Please make sure a config.json exists (or model_index.json for Stable "
            f"Diffusion Models) for this model in the huggingface hub"
        )
    return model_config


def _create_estimator(
    instance_type: str,
    s3_output_uri: str,
    image_uri: str,
    role: str,
    sagemaker_session: Optional[Session],
    volume_size: int,
    vpc_config: Optional[
        Dict[
            str,
            List[
                str,
            ],
        ]
    ] = None,
    volume_kms_key=None,
    output_kms_key=None,
    use_spot_instances: bool = False,
    max_wait: int = None,
    enable_network_isolation: bool = False,
):
    """Placeholder docstring"""

    subnets = None
    security_group_ids = None
    if vpc_config:
        subnets = vpc_config.get("Subnets")
        security_group_ids = vpc_config.get("SecurityGroupIds")

    return Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        volume_size=volume_size,
        volume_kms_key=volume_kms_key,
        output_path=s3_output_uri,
        output_kms_key=output_kms_key,
        sagemaker_session=sagemaker_session,
        subnets=subnets,
        security_group_ids=security_group_ids,
        use_spot_instances=use_spot_instances,
        max_wait=max_wait,
        enable_network_isolation=enable_network_isolation,
    )


class DJLModel(FrameworkModel):
    """A DJL SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``."""

    def __new__(
        cls,
        model_id: str,
        *args,
        **kwargs,
    ):  # pylint: disable=W0613
        """Create a specific subclass of DJLModel for a given engine"""

        if model_id.endswith("tar.gz"):
            raise ValueError(
                "DJLModel does not support model artifacts in tar.gz format."
                "Please store the model in uncompressed format and provide the s3 uri of the "
                "containing folder"
            )
        if model_id.startswith("s3://"):
            sagemaker_session = kwargs.get("sagemaker_session")
            model_config = _get_model_config_properties_from_s3(model_id, sagemaker_session)
        else:
            model_config = _get_model_config_properties_from_hf(model_id)
        if model_config.get("_class_name") == "StableDiffusionPipeline":
            model_type = defaults.STABLE_DIFFUSION_MODEL_TYPE
            num_heads = 0
        else:
            model_type = model_config.get("model_type")
            num_heads = model_config.get("n_head") or model_config.get("num_attention_heads")
        number_of_partitions = kwargs.get("number_of_partitions") or kwargs.get(
            "tensor_parallel_degree"
        )
        cls_to_create = (
            _validate_engine_for_model_type(cls, model_type, number_of_partitions, num_heads)
            if cls is not DJLModel
            else _determine_engine_for_model(model_type, number_of_partitions, num_heads)
        )
        instance = super().__new__(cls_to_create)
        if model_type == defaults.STABLE_DIFFUSION_MODEL_TYPE:
            instance.engine = DJLServingEngineEntryPointDefaults.STABLE_DIFFUSION
        elif isinstance(instance, DeepSpeedModel):
            instance.engine = DJLServingEngineEntryPointDefaults.DEEPSPEED
        elif isinstance(instance, FasterTransformerModel):
            instance.engine = DJLServingEngineEntryPointDefaults.FASTER_TRANSFORMER
        else:
            instance.engine = DJLServingEngineEntryPointDefaults.HUGGINGFACE_ACCELERATE
        return instance

    def __init__(
        self,
        model_id: str,
        role: str,
        djl_version: Optional[str] = None,
        task: Optional[str] = None,
        dtype: str = "fp32",
        number_of_partitions: Optional[int] = None,
        min_workers: Optional[int] = None,
        max_workers: Optional[int] = None,
        job_queue_size: Optional[int] = None,
        parallel_loading: bool = False,
        model_loading_timeout: Optional[int] = None,
        prediction_timeout: Optional[int] = None,
        entry_point: Optional[str] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        predictor_cls: callable = DJLPredictor,
        **kwargs,
    ):
        """Initialize a DJLModel.

        Args:
            model_id (str): This is either the HuggingFace Hub model_id, or the Amazon S3 location
                containing the uncompressed model artifacts (i.e. not a tar.gz file).
                The model artifacts are expected to be in HuggingFace pre-trained model
                format (i.e. model should be loadable from the huggingface transformers
                from_pretrained api, and should also include tokenizer configs if applicable).
            role (str): An AWS IAM role specified with either the name or full ARN. The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access model artifacts. After the endpoint is created,
                the inference code might use the IAM role, if it needs to access an AWS resource.
            djl_version (str): DJL Serving version you want to use for serving your model for
                inference. Defaults to None. If not provided, the latest available version of DJL
                Serving is used. This is not used if ``image_uri`` is provided.
            task (str): The HuggingFace/NLP task you want to launch this model for. Defaults to
                None.
                If not provided, the task will be inferred from the model architecture by DJL.
            dtype (str): The data type to use for loading your model. Accepted values are
                "fp32", "fp16", "bf16", "int8". Defaults to "fp32".
            number_of_partitions (int): The number of GPUs to partition the model across. The
                partitioning strategy is determined by the selected backend. If DeepSpeed is
                selected, this is tensor parallelism.
                If HuggingFace Accelerate is selected, this is a naive sharding strategy
                that splits the model layers across the available resources. Defaults to None. If
                not provided, no model partitioning is done.
            min_workers (int): The minimum number of worker processes. Defaults to None. If not
                provided, dJL Serving will automatically detect the minimum workers.
            max_workers (int): The maximum number of worker processes. Defaults to None. If not
                provided, DJL Serving will automatically detect the maximum workers.
            job_queue_size (int): The request job queue size. Defaults to None. If not specified,
                defaults to 1000.
            parallel_loading (bool): Whether to load model workers in parallel. Defaults to False,
                in which case DJL Serving will load the model workers sequentially to reduce the
                risk of running out of memory. Set to True if you want to reduce model loading
                time and know that peak memory usage will not cause out of memory issues.
            model_loading_timeout (int): The worker model loading timeout in seconds. Defaults to
                None. If not provided, the default is 240 seconds.
            prediction_timeout (int): The worker predict call (handler) timeout in seconds.
                Defaults to None. If not provided, the default is 120 seconds.
            entry_point (str): This can either be the absolute or relative path to the Python source
                file that should be executed as the entry point to model
                hosting, or a python module that is installed in the container. If ``source_dir``
                is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``. Defaults to None.
            image_uri (str): A docker image URI. Defaults to None. If not specified, a default
                image for DJL Serving will be used based on ``djl_version``. If ``djl_version``
                is not specified, the latest available container version will be used.
            predictor_cls (callable[str, sagemaker.session.Session]): A function to call to create a
                predictor with an endpoint name and SageMaker ``Session``. If specified,
                ``deploy()`` returns
                the result of invoking this function on the created endpoint name.
            **kwargs: Keyword arguments passed to the superclass
                :class:`~sagemaker.model.FrameworkModel` and, subsequently, its
                superclass :class:`~sagemaker.model.Model`.

        .. tip::

            Instantiating a DJLModel will return an instance of either
            :class:`~sagemaker.djl_inference.DeepSpeedModel` or
            :class:`~sagemaker.djl_inference.HuggingFaceAccelerateModel` based on our framework
            recommendation for the model type.

            If you want to use a specific framework to deploy your model with, we recommend
            instantiating that specific
            model class directly. The available framework specific classes are
            :class:`~sagemaker.djl_inference.DeepSpeedModel` or
            :class:`~sagemaker.djl_inference.HuggingFaceAccelerateModel`
        """

        if kwargs.get("model_data"):
            logger.warning(
                "DJLModels do not use model_data parameter. model_data parameter will be ignored."
                "You only need to set model_id and ensure it points to uncompressed model "
                "artifacts in s3, or a valid HuggingFace Hub model_id."
            )
        data_type = kwargs.pop("data_type", None)
        if data_type:
            logger.warning(
                "data_type is being deprecated in favor of dtype. Please migrate use of data_type"
                " to dtype. Support for data_type will be removed in a future release"
            )
            dtype = dtype or data_type
        super(DJLModel, self).__init__(
            None, image_uri, role, entry_point, predictor_cls=predictor_cls, **kwargs
        )
        self.model_id = model_id
        self.djl_version = djl_version
        self.task = task
        self.dtype = dtype
        self.number_of_partitions = number_of_partitions
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.job_queue_size = job_queue_size
        self.parallel_loading = parallel_loading
        self.model_loading_timeout = model_loading_timeout
        self.prediction_timeout = prediction_timeout
        self.sagemaker_session = self.sagemaker_session or Session()
        self.save_mp_checkpoint_path = None

    def package_for_edge(self, **_):
        """Not implemented.

        DJLModels do not support SageMaker edge.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("DJLModels do not support Sagemaker Edge")

    def compile(self, **_):
        """Not implemented.

        DJLModels do not support SageMaker Neo compilation.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "DJLModels do not currently support compilation with SageMaker Neo"
        )

    def transformer(self, **_):
        """Not implemented.

        DJLModels do not support SageMaker Batch Transform.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "DJLModels do not currently support Batch Transform inference jobs"
        )

    def right_size(self, **_):
        """Not implemented.

        DJLModels do not support SageMaker Inference Recommendation Jobs.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "DJLModels do not currently support Inference Recommendation Jobs"
        )

    def partition(
        self,
        instance_type: str,
        s3_output_uri: str = None,
        s3_output_prefix: str = "aot-partitioned-checkpoints",
        job_name: Optional[str] = None,
        volume_size: int = 30,
        volume_kms_key: Optional[str] = None,
        output_kms_key: Optional[str] = None,
        use_spot_instances: bool = False,
        max_wait: int = None,
        enable_network_isolation: bool = False,
    ):
        """Partitions the model using SageMaker Training Job. This is a synchronous API call.

        Args:
            instance_type (str): The EC2 instance type to partition this Model.
                    For example, 'ml.p4d.24xlarge'.
            s3_output_uri (str): S3 location for saving the training result (model
                    artifacts and output files). If not specified, results are
                    stored to a default bucket. If the bucket with the specific name
                    does not exist, it will be created.
            s3_output_prefix (str): Name of the prefix where all the partitioned
                    checkpoints to be uploaded. If not provided, the default value is
                    aot-partitioned-checkpoints.
            job_name (str): Training job name. If not specified, a unique training job
                        name will be created.
            volume_size (int): Size in GB of the storage volume to use for
                storing input and output data during training (default: 30).
            volume_kms_key (str): Optional. KMS key ID for encrypting EBS
                volume attached to the training instance (default: None).
            output_kms_key (str): Optional. KMS key ID for encrypting the
                training output (default: None).
            use_spot_instances (bool): Specifies whether to use SageMaker
                Managed Spot instances for training. If enabled then the
                ``max_wait`` arg should also be set.

                More information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
                (default: ``False``).
            max_wait (int): Timeout in seconds waiting for spot training
                job (default: None). After this amount of time Amazon
                SageMaker will stop waiting for managed spot training job to
                complete (default: None).
            enable_network_isolation (bool): Specifies whether container will
                run in network isolation mode (default: ``False``). Network
                isolation mode restricts the container access to outside networks
                (such as the Internet). The container does not make any inbound or
                outbound network calls. Also known as Internet-free mode.
        Returns:
            None
        """

        if not self.image_uri:
            region_name = self.sagemaker_session.boto_session.region_name
            self.image_uri = self.serving_image_uri(region_name)

        if s3_output_uri is None:
            deploy_key_prefix = fw_utils.model_code_key_prefix(
                self.key_prefix, self.name, self.image_uri
            )

            bucket, deploy_key_prefix = s3.determine_bucket_and_prefix(
                bucket=self.bucket,
                key_prefix=deploy_key_prefix,
                sagemaker_session=self.sagemaker_session,
            )
            s3_output_uri = s3_path_join("s3://", bucket, deploy_key_prefix)

        self.save_mp_checkpoint_path = s3_path_join(s3_output_uri, s3_output_prefix)

        container_def = self._upload_model_to_s3(upload_as_tar=False)
        estimator = _create_estimator(
            instance_type=instance_type,
            s3_output_uri=s3_output_uri,
            image_uri=self.image_uri,
            role=self.role,
            sagemaker_session=self.sagemaker_session,
            volume_size=volume_size,
            vpc_config=self.vpc_config,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            use_spot_instances=use_spot_instances,
            max_wait=max_wait,
            enable_network_isolation=enable_network_isolation,
        )

        # creates a training job to do partitions
        estimator.fit(
            inputs=container_def["ModelDataUrl"],
            wait=True,
            logs="All",
            job_name=job_name,
            experiment_config=None,
        )

        self.model_id = self.save_mp_checkpoint_path
        # reset save_mp_checkpoint_path since partition is completed.
        self.save_mp_checkpoint_path = None

    def deploy(
        self,
        instance_type,
        initial_instance_count=1,
        serializer=None,
        deserializer=None,
        endpoint_name=None,
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
    ):
        """Deploy this ``Model`` to an ``Endpoint`` and optionally return a ``Predictor``.

        Create a SageMaker ``Model`` and ``EndpointConfig``, and deploy an
        ``Endpoint`` from this ``Model``. If ``self.predictor_cls`` is not None,
        this method returns the result of invoking ``self.predictor_cls`` on
        the created endpoint name.

        The name of the created model is accessible in the ``name`` field of
        this ``Model`` after deploy returns

        The name of the created endpoint is accessible in the
        ``endpoint_name`` field of this ``Model`` after deploy returns.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p4d.24xlarge'.
            initial_instance_count (int): The initial number of instances to run
                in the ``Endpoint`` created from this ``Model``. It needs to be at least 1 (
                default: 1)
            serializer (:class:`~sagemaker.serializers.BaseSerializer`): A
                serializer object, used to encode data for an inference endpoint
                (default: None). If ``serializer`` is not None, then
                ``serializer`` will override the default serializer. The
                default serializer is set by the ``predictor_cls``.
            deserializer (:class:`~sagemaker.deserializers.BaseDeserializer`): A
                deserializer object, used to decode data from an inference
                endpoint (default: None). If ``deserializer`` is not None, then
                ``deserializer`` will override the default deserializer. The
                default deserializer is set by the ``predictor_cls``.
            endpoint_name (str): The name of the endpoint to create (default:
                None). If not specified, a unique endpoint name will be created.
            tags (List[dict[str, str]]): The list of tags to attach to this
                specific endpoint.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                data on the storage volume attached to the instance hosting the
                endpoint.
            wait (bool): Whether the call should wait until the deployment of
                this model completes (default: True).
            data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. Default: None.
            volume_size (int): The size, in GB, of the ML storage volume attached to individual
                inference instance associated with the production variant. Currenly only Amazon EBS
                gp2 storage volumes are supported.
            model_data_download_timeout (int): The timeout value, in seconds, to download and
                extract model data from Amazon S3 to the individual inference instance associated
                with this production variant.
            container_startup_health_check_timeout (int): The timeout value, in seconds, for your
                inference container to pass health check by SageMaker Hosting. For more information
                about health check see:
                https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code
                .html#your-algorithms-inference-algo-ping-requests

        Returns:
            callable[string, sagemaker.session.Session] or None: Invocation of
                ``self.predictor_cls`` on the created endpoint name, if ``self.predictor_cls``
                is not None. Otherwise, return None.
        """

        instance_family = instance_type.rsplit(".", 1)[0]
        if instance_family not in defaults.ALLOWED_INSTANCE_FAMILIES:
            raise ValueError(
                f"Invalid instance type. DJLModels only support deployment to instances"
                f"with GPUs. Supported instance families are {defaults.ALLOWED_INSTANCE_FAMILIES}"
            )

        return super(DJLModel, self).deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            serializer=serializer,
            deserializer=deserializer,
            endpoint_name=endpoint_name,
            tags=tags,
            kms_key=kms_key,
            wait=wait,
            data_capture_config=data_capture_config,
            volume_size=volume_size,
            model_data_download_timeout=model_data_download_timeout,
            container_startup_health_check_timeout=container_startup_health_check_timeout,
        )

    def _upload_model_to_s3(self, upload_as_tar: bool = True):
        """Placeholder docstring"""

        if not self.image_uri:
            region_name = self.sagemaker_session.boto_session.region_name
            self.image_uri = self.serving_image_uri(region_name)

        environment = self._get_container_env()

        local_download_dir = (
            None
            if self.sagemaker_session.settings is None
            or self.sagemaker_session.settings.local_download_dir is None
            else self.sagemaker_session.settings.local_download_dir
        )
        with _tmpdir(directory=local_download_dir) as tmp:
            if self.source_dir or self.entry_point:
                # Below method downloads from s3, or moves local files to tmp/code
                _create_or_update_code_dir(
                    tmp,
                    self.entry_point,
                    self.source_dir,
                    self.dependencies,
                    self.sagemaker_session,
                    tmp,
                )
            tmp_code_dir = os.path.join(tmp, "code")
            existing_serving_properties = _read_existing_serving_properties(tmp_code_dir)
            kwargs_serving_properties = self.generate_serving_properties()
            existing_serving_properties.update(kwargs_serving_properties)

            if not os.path.exists(tmp_code_dir):
                os.mkdir(tmp_code_dir)
            with open(os.path.join(tmp_code_dir, "serving.properties"), "w+") as f:
                for key, val in existing_serving_properties.items():
                    f.write(f"{key}={val}\n")

            deploy_key_prefix = fw_utils.model_code_key_prefix(
                self.key_prefix, self.name, self.image_uri
            )
            bucket, deploy_key_prefix = s3.determine_bucket_and_prefix(
                bucket=self.bucket,
                key_prefix=deploy_key_prefix,
                sagemaker_session=self.sagemaker_session,
            )
            if upload_as_tar:
                uploaded_code = fw_utils.tar_and_upload_dir(
                    self.sagemaker_session.boto_session,
                    bucket,
                    deploy_key_prefix,
                    self.entry_point,
                    directory=tmp_code_dir,
                    dependencies=self.dependencies,
                    kms_key=self.model_kms_key,
                )
                model_data_url = uploaded_code.s3_prefix
            else:
                model_data_url = S3Uploader.upload(
                    tmp_code_dir,
                    s3_path_join("s3://", bucket, deploy_key_prefix, "aot-model"),
                    self.model_kms_key,
                    self.sagemaker_session,
                )
            return sagemaker.container_def(
                self.image_uri, model_data_url=model_data_url, env=environment
            )

    def prepare_container_def(
        self,
        instance_type=None,
        accelerator_type=None,
        serverless_inference_config=None,
    ):  # pylint: disable=unused-argument
        """A container definition with framework configuration set in model environment variables.

        Returns:
            dict[str, str]: A container definition object usable with the
            CreateModel API.
        """

        return self._upload_model_to_s3(upload_as_tar=True)

    def generate_serving_properties(self, serving_properties=None) -> Dict[str, str]:
        """Generates the DJL Serving configuration to use for the model.

        The configuration is generated using the arguments passed to the Model during
        initialization. If a serving.properties file is found in ``self.source_dir``,
        those configuration as merged with the Model parameters, with Model parameters taking
        priority.

        Args:
            serving_properties: Dictionary containing existing model server configuration
            obtained from ``self.source_dir``. Defaults to None.

        Returns:
            dict: The model server configuration to use when deploying this model to SageMaker.
        """
        if not serving_properties:
            serving_properties = {}
        serving_properties["engine"] = self.engine.value[0]  # pylint: disable=E1101
        serving_properties["option.entryPoint"] = self.engine.value[1]  # pylint: disable=E1101
        serving_properties["option.model_id"] = self.model_id
        if self.number_of_partitions:
            serving_properties["option.tensor_parallel_degree"] = self.number_of_partitions
        if self.entry_point:
            serving_properties["option.entryPoint"] = self.entry_point
        if self.task:
            serving_properties["option.task"] = self.task
        if self.dtype:
            serving_properties["option.dtype"] = self.dtype
        if self.min_workers:
            serving_properties["minWorkers"] = self.min_workers
        if self.max_workers:
            serving_properties["maxWorkers"] = self.max_workers
        if self.job_queue_size:
            serving_properties["job_queue_size"] = self.job_queue_size
        if self.parallel_loading:
            serving_properties["option.parallel_loading"] = self.parallel_loading
        if self.model_loading_timeout:
            serving_properties["option.model_loading_timeout"] = self.model_loading_timeout
        if self.prediction_timeout:
            serving_properties["option.prediction_timeout"] = self.prediction_timeout
        if self.save_mp_checkpoint_path:
            serving_properties["option.save_mp_checkpoint_path"] = self.save_mp_checkpoint_path
        return serving_properties

    def serving_image_uri(self, region_name):
        """Create a URI for the serving image.

        Args:
            region_name (str): AWS region where the image is uploaded.

        Returns:
            str: The appropriate image URI based on the given parameters.
        """
        if not self.djl_version:
            self.djl_version = "0.23.0"

        return image_uris.retrieve(
            self._framework(),
            region_name,
            version=self.djl_version,
        )

    def _get_container_env(self):
        """Placeholder docstring"""

        if not self.container_log_level:
            return self.env

        if self.container_log_level not in _LOG_LEVEL_MAP:
            logger.warning("Ignoring invalid container log level: %s", self.container_log_level)
            return self.env

        self.env[
            "SERVING_OPTS"
        ] = f'"-Dai.djl.logging.level={_LOG_LEVEL_MAP[self.container_log_level]}"'
        return self.env


class DeepSpeedModel(DJLModel):
    """A DJL DeepSpeed SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``"""

    _framework_name = "djl-deepspeed"

    def __init__(
        self,
        model_id: str,
        role: str,
        tensor_parallel_degree: Optional[int] = None,
        max_tokens: Optional[int] = None,
        low_cpu_mem_usage: bool = False,
        enable_cuda_graph: bool = False,
        triangular_masking: bool = True,
        return_tuple: bool = True,
        **kwargs,
    ):
        """Initialize a DeepSpeedModel

        Args:
            model_id (str): This is either the HuggingFace Hub model_id, or the Amazon S3 location
                containing the uncompressed model artifacts (i.e. not a tar.gz file).
                The model artifacts are expected to be in HuggingFace pre-trained model
                format (i.e. model should be loadable from the huggingface transformers
                from_pretrained api, and should also include tokenizer configs if applicable).
            role (str): An AWS IAM role specified with either the name or full ARN. The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access model artifacts. After the endpoint is created,
                the inference code
                might use the IAM role, if it needs to access an AWS resource.
            tensor_parallel_degree (int): The number of gpus to shard a single instance of the
                 model across via tensor_parallelism. This should be set to greater than 1 if the
                 size of the model is larger than the memory available on a single GPU on the
                 instance. Defaults to None. If not set, no tensor parallel sharding is done.
            max_tokens (int): The maximum number of tokens (input + output tokens) the DeepSpeed
                engine is configured for. Defaults to None. If not set, the DeepSpeed default of
                1024 is used.
            low_cpu_mem_usage (bool): Whether to limit CPU memory usage to 1x model size during
                model loading. This is an experimental feature in HuggingFace. This is useful when
                loading multiple instances of your model in parallel. Defaults to False.
            enable_cuda_graph (bool): Whether to enable CUDA graph replay to accelerate inference
                passes. This cannot be used with tensor parallelism greater than 1.
                Defaults to False.
            triangular_masking (bool): Whether to use triangular attention mask. This is
                application specific. Defaults to True.
            return_tuple (bool): Whether the transformer layers need to return a tuple or a
                Tensor. Defaults to True.
            **kwargs: Keyword arguments passed to the superclasses
                :class:`~sagemaker.djl_inference.DJLModel`,
                :class:`~sagemaker.model.FrameworkModel`, and
                :class:`~sagemaker.model.Model`

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.djl_inference.DJLModel`,
            :class:`~sagemaker.model.FrameworkModel`, and
            :class:`~sagemaker.model.Model`.
        """

        super(DeepSpeedModel, self).__init__(
            model_id,
            role,
            **kwargs,
        )
        if self.number_of_partitions and tensor_parallel_degree:
            logger.warning(
                "Both number_of_partitions and tensor_parallel_degree have been set for "
                "DeepSpeedModel."
                "These mean the same thing for DeepSpeedModel. Please only set "
                "tensor_parallel_degree."
                "number_of_partitions will be ignored"
            )
        self.number_of_partitions = tensor_parallel_degree or self.number_of_partitions
        self.max_tokens = max_tokens
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.enable_cuda_graph = enable_cuda_graph
        self.triangular_masking = triangular_masking
        self.return_tuple = return_tuple
        self.save_mp_checkpoint_path = None
        self.checkpoint = None

    def generate_serving_properties(self, serving_properties=None) -> Dict[str, Any]:
        """Generates the DJL Serving configuration to use for the model.

        The configuration is generated using the arguments passed to the Model during
        initialization. If a serving.properties file is found in ``self.source_dir``,
        those configuration as merged with the Model parameters, with Model parameters taking
        priority.

        Args:
            serving_properties: Dictionary containing existing model server configuration
            obtained from ``self.source_dir``. Defaults to None.

        Returns:
            dict: The model server configuration to use when deploying this model to SageMaker.
        """

        serving_properties = super(DeepSpeedModel, self).generate_serving_properties(
            serving_properties=serving_properties
        )
        if self.max_tokens:
            serving_properties["option.max_tokens"] = self.max_tokens
        if self.low_cpu_mem_usage:
            serving_properties["option.low_cpu_mem_usage"] = self.low_cpu_mem_usage
        if self.enable_cuda_graph:
            if self.number_of_partitions > 1:
                raise ValueError(
                    "enable_cuda_graph is not supported when tensor_parallel_degree > 1"
                )
            serving_properties["option.enable_cuda_graph"] = self.enable_cuda_graph
        if self.triangular_masking:
            serving_properties["option.triangular_masking"] = self.triangular_masking
        if self.return_tuple:
            serving_properties["option.return_tuple"] = self.return_tuple
        if self.save_mp_checkpoint_path:
            serving_properties["option.save_mp_checkpoint_path"] = self.save_mp_checkpoint_path
        if self.checkpoint:
            serving_properties["option.checkpoint"] = self.checkpoint

        return serving_properties

    def partition(
        self,
        instance_type: str,
        s3_output_uri: str = None,
        s3_output_prefix: str = "aot-partitioned-checkpoints",
        job_name: Optional[str] = None,
        volume_size: int = 30,
        volume_kms_key: Optional[str] = None,
        output_kms_key: Optional[str] = None,
        use_spot_instances: bool = False,
        max_wait: int = None,
        enable_network_isolation: bool = False,
    ):
        """Partitions the model using SageMaker Training Job. This is a synchronous API call.

        Args:
            instance_type (str): The EC2 instance type to partition this Model.
                    For example, 'ml.p4d.24xlarge'.
            s3_output_uri (str): S3 location for saving the training result (model
                    artifacts and output files). If not specified, results are
                    stored to a default bucket. If the bucket with the specific name
                    does not exist, it will be created.
            s3_output_prefix (str): Name of the prefix where all the partitioned
                    checkpoints to be uploaded. If not provided, the default value is
                    aot-partitioned-checkpoints.
            job_name (str): Training job name. If not specified, a unique training job
                        name will be created.
            volume_size (int): Size in GB of the storage volume to use for
                storing input and output data during training (default: 30).
            volume_kms_key (str): Optional. KMS key ID for encrypting EBS
                volume attached to the training instance (default: None).
            output_kms_key (str): Optional. KMS key ID for encrypting the
                training output (default: None).
            use_spot_instances (bool): Specifies whether to use SageMaker
                Managed Spot instances for training. If enabled then the
                ``max_wait`` arg should also be set.

                More information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
                (default: ``False``).
            max_wait (int): Timeout in seconds waiting for spot training
                job (default: None). After this amount of time Amazon
                SageMaker will stop waiting for managed spot training job to
                complete (default: None).
            enable_network_isolation (bool): Specifies whether container will
                run in network isolation mode (default: ``False``). Network
                isolation mode restricts the container access to outside networks
                (such as the Internet). The container does not make any inbound or
                outbound network calls. Also known as Internet-free mode.
        Returns:
            None
        """

        super(DeepSpeedModel, self).partition(
            instance_type,
            s3_output_uri,
            s3_output_prefix=s3_output_prefix,
            job_name=job_name,
            volume_size=volume_size,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            use_spot_instances=use_spot_instances,
            max_wait=max_wait,
            enable_network_isolation=enable_network_isolation,
        )

        self.checkpoint = "ds_inference_config.json"


class HuggingFaceAccelerateModel(DJLModel):
    """A DJL Hugging Face SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``."""

    _framework_name = "djl-deepspeed"

    def __init__(
        self,
        model_id: str,
        role: str,
        number_of_partitions: Optional[int] = None,
        device_id: Optional[int] = None,
        device_map: Optional[Union[str, Dict[str, str]]] = None,
        load_in_8bit: bool = False,
        low_cpu_mem_usage: bool = False,
        **kwargs,
    ):
        """Initialize a HuggingFaceAccelerateModel.

        Args:
            model_id (str): This is either the HuggingFace Hub model_id, or the Amazon S3 location
                containing the uncompressed model artifacts (i.e. not a tar.gz file).
                The model artifacts are expected to be in HuggingFace pre-trained model
                format (i.e. model should be loadable from the huggingface transformers
                from_pretrained api, and should also include tokenizer configs if applicable).
            role (str): An AWS IAM role specified with either the name or full ARN. The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access model artifacts. After the endpoint is created,
                the inference code
                might use the IAM role, if it needs to access an AWS resource.
            number_of_partitions (int): The number of GPUs to partition the model across. The
                partitioning strategy is determined by the device_map setting. If device_map is
                not specified, the default HuggingFace strategy will be used.
            device_id (int): The device_id to use for instantiating the model. If provided,
                the model will only be instantiated once on the indicated device. Do not set this
                if you have also specified data_parallel_degree. Defaults to None.
            device_map (str or dict): The HuggingFace accelerate device_map to use. Defaults to
                None.
            load_in_8bit (bool): Whether to load the model in int8 precision using bits and bytes
                quantization. This is only supported for select model architectures.
                Defaults to False. If ``dtype`` is int8, then this is set to True.
            low_cpu_mem_usage (bool): Whether to limit CPU memory usage to 1x model size during
                model loading. This is an experimental feature in HuggingFace. This is useful when
                loading multiple instances of your model in parallel. Defaults to False.
            **kwargs: Keyword arguments passed to the superclasses
                :class:`~sagemaker.djl_inference.DJLModel`,
                :class:`~sagemaker.model.FrameworkModel`, and
                :class:`~sagemaker.model.Model`

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.djl_inference.DJLModel`,
            :class:`~sagemaker.model.FrameworkModel`, and
            :class:`~sagemaker.model.Model`.
        """

        super(HuggingFaceAccelerateModel, self).__init__(
            model_id,
            role,
            number_of_partitions=number_of_partitions,
            **kwargs,
        )
        self.device_id = device_id
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.low_cpu_mem_usage = low_cpu_mem_usage

    def generate_serving_properties(self, serving_properties=None) -> Dict[str, str]:
        """Generates the DJL Serving configuration to use for the model.

        The configuration is generated using the arguments passed to the Model during
        initialization. If a serving.properties file is found in ``self.source_dir``,
        those configuration as merged with the Model parameters, with Model parameters taking
        priority.

        Args:
            serving_properties: Dictionary containing existing model server configuration
            obtained from ``self.source_dir``. Defaults to None.

        Returns:
            dict: The model server configuration to use when deploying this model to SageMaker.
        """
        serving_properties = super(HuggingFaceAccelerateModel, self).generate_serving_properties(
            serving_properties=serving_properties
        )
        if self.device_id:
            if self.number_of_partitions > 1:
                raise ValueError("device_id cannot be set when number_of_partitions is > 1")
            serving_properties["option.device_id"] = self.device_id
        if self.device_map:
            serving_properties["option.device_map"] = self.device_map
        if self.load_in_8bit:
            if self.dtype != "int8":
                raise ValueError("Set dtype='int8' to use load_in_8bit")
            serving_properties["option.load_in_8bit"] = self.load_in_8bit
        if self.dtype == "int8":
            serving_properties["option.load_in_8bit"] = True
        if self.low_cpu_mem_usage:
            serving_properties["option.low_cpu_mem_usage"] = self.low_cpu_mem_usage
        # This is a workaround due to a bug in our built in handler for huggingface
        # TODO: Remove this logic whenever 0.20.0 image is out of service
        if (
            serving_properties["option.entryPoint"] == "djl_python.huggingface"
            and self.dtype
            and self.dtype != "auto"
            and self.djl_version
            and int(self.djl_version.split(".")[1]) < 21
        ):
            serving_properties["option.dtype"] = "auto"
            serving_properties.pop("option.load_in_8bit", None)
        return serving_properties

    def partition(
        self,
        instance_type: str,
        s3_output_uri: str = None,
        s3_output_prefix: str = "aot-partitioned-checkpoints",
        job_name: Optional[str] = None,
        volume_size: int = 30,
        volume_kms_key: Optional[str] = None,
        output_kms_key: Optional[str] = None,
        use_spot_instances: bool = False,
        max_wait: int = None,
        enable_network_isolation: bool = False,
    ):
        """Partitions the model using SageMaker Training Job. This is a synchronous API call.

        Args:
            instance_type (str): The EC2 instance type to partition this Model.
                    For example, 'ml.p4d.24xlarge'.
            s3_output_uri (str): S3 location for saving the training result (model
                    artifacts and output files). If not specified, results are
                    stored to a default bucket. If the bucket with the specific name
                    does not exist, it will be created.
            s3_output_prefix (str): Name of the prefix where all the partitioned
                    checkpoints to be uploaded. If not provided, the default value is
                    aot-partitioned-checkpoints.
            job_name (str): Training job name. If not specified, a unique training job
                        name will be created.
            volume_size (int): Size in GB of the storage volume to use for
                storing input and output data during training (default: 30).
            volume_kms_key (str): Optional. KMS key ID for encrypting EBS
                volume attached to the training instance (default: None).
            output_kms_key (str): Optional. KMS key ID for encrypting the
                training output (default: None).
            use_spot_instances (bool): Specifies whether to use SageMaker
                Managed Spot instances for training. If enabled then the
                ``max_wait`` arg should also be set.

                More information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
                (default: ``False``).
            max_wait (int): Timeout in seconds waiting for spot training
                job (default: None). After this amount of time Amazon
                SageMaker will stop waiting for managed spot training job to
                complete (default: None).
            enable_network_isolation (bool): Specifies whether container will
                run in network isolation mode (default: ``False``). Network
                isolation mode restricts the container access to outside networks
                (such as the Internet). The container does not make any inbound or
                outbound network calls. Also known as Internet-free mode.
        Returns:
            None
        """

        logger.warning(
            "HuggingFace engine does not currently support tensor parallelism. "
            "Hence ahead of time partitioning is skipped"
        )


class FasterTransformerModel(DJLModel):
    """A DJL FasterTransformer SageMaker ``Model``

    This can be deployed to a SageMaker ``Endpoint``.
    """

    _framework_name = "djl-fastertransformer"

    def __init__(
        self,
        model_id: str,
        role: str,
        tensor_parallel_degree: Optional[int] = None,
        **kwargs,
    ):
        """Initialize a FasterTransformerModel.

        Args:
            model_id (str): This is either the HuggingFace Hub model_id, or the Amazon S3 location
                containing the uncompressed model artifacts (i.e. not a tar.gz file).
                The model artifacts are expected to be in HuggingFace pre-trained model
                format (i.e. model should be loadable from the huggingface transformers
                from_pretrained api, and should also include tokenizer configs if applicable).
            role (str): An AWS IAM role specified with either the name or full ARN. The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access model artifacts. After the endpoint is created,
                the inference code
                might use the IAM role, if it needs to access an AWS resource.
            tensor_parllel_degree (int): The number of gpus to shard a single instance of the
                 model across via tensor_parallelism. This should be set to greater than 1 if the
                 size of the model is larger than the memory available on a single GPU on the
                 instance. Defaults to None. If not set, no tensor parallel sharding is done.
            **kwargs: Keyword arguments passed to the superclasses
                :class:`~sagemaker.djl_inference.DJLModel`,
                :class:`~sagemaker.model.FrameworkModel`, and
                :class:`~sagemaker.model.Model`

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.djl_inference.DJLModel`,
            :class:`~sagemaker.model.FrameworkModel`, and
            :class:`~sagemaker.model.Model`.
        """

        super(FasterTransformerModel, self).__init__(
            model_id,
            role,
            **kwargs,
        )
        if self.number_of_partitions and tensor_parallel_degree:
            logger.warning(
                "Both number_of_partitions and tensor_parallel_degree have been set for "
                "FasterTransformerModel."
                "These mean the same thing for FasterTransformerModel. Please only set "
                "tensor_parallel_degree."
                "number_of_partitions will be ignored"
            )
        self.number_of_partitions = tensor_parallel_degree or self.number_of_partitions
