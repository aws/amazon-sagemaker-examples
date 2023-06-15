import copy
import os

import sagemaker
from sagemaker import fw_utils, image_uris, vpc_utils
from sagemaker.estimator import Estimator
from sagemaker.model import DIR_PARAM_NAME, SCRIPT_PARAM_NAME, Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer, NumpySerializer

from deserializers import PandasDeserializer
from sagemaker_utils import retrieve_latest_framework_version
from serializers import MultiModalSerializer, ParquetSerializer


# Estimator documentation: https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#estimators
class AutoGluonSagemakerEstimator(Estimator):
    def __init__(
        self,
        entry_point,
        region,
        framework_version,
        py_version,
        instance_type,
        source_dir=None,
        hyperparameters=None,
        custom_image_uri=None,
        **kwargs,
    ):
        self.framework_version = framework_version
        self.py_version = py_version
        self.image_uri = custom_image_uri
        if self.image_uri is None:
            self.image_uri = image_uris.retrieve(
                "autogluon",
                region=region,
                version=framework_version,
                py_version=py_version,
                image_scope="training",
                instance_type=instance_type,
            )
        super().__init__(
            entry_point=entry_point,
            source_dir=source_dir,
            hyperparameters=hyperparameters,
            instance_type=instance_type,
            image_uri=self.image_uri,
            **kwargs,
        )

    def _configure_distribution(self, distributions):
        return

    def create_model(
        self,
        region,
        framework_version,
        py_version,
        instance_type,
        source_dir=None,
        entry_point=None,
        role=None,
        image_uri=None,
        predictor_cls=None,
        vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
        repack=False,
        **kwargs,
    ):
        image_uri = image_uris.retrieve(
            "autogluon",
            region=region,
            version=framework_version,
            py_version=py_version,
            image_scope="inference",
            instance_type=instance_type,
        )
        if predictor_cls is None:

            def predict_wrapper(endpoint, session):
                return Predictor(endpoint, session)

            predictor_cls = predict_wrapper

        role = role or self.role

        if "enable_network_isolation" not in kwargs:
            kwargs["enable_network_isolation"] = self.enable_network_isolation()

        if repack:
            model_cls = AutoGluonRepackInferenceModel
        else:
            model_cls = AutoGluonNonRepackInferenceModel
        return model_cls(
            image_uri=image_uri,
            source_dir=source_dir,
            entry_point=entry_point,
            model_data=self.model_data,
            role=role,
            vpc_config=self.get_vpc_config(vpc_config_override),
            sagemaker_session=self.sagemaker_session,
            predictor_cls=predictor_cls,
            **kwargs,
        )

    @classmethod
    def _prepare_init_params_from_job_description(
        cls, job_details, model_channel_name=None
    ):
        init_params = super()._prepare_init_params_from_job_description(
            job_details, model_channel_name=model_channel_name
        )
        # This two parameters will not be used, but is required to reattach the job
        init_params["region"] = "us-east-1"
        init_params["framework_version"] = retrieve_latest_framework_version()
        return init_params


# Documentation for Model: https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#model
class AutoGluonSagemakerInferenceModel(Model):
    def __init__(
        self,
        model_data,
        role,
        entry_point,
        region,
        framework_version,
        py_version,
        instance_type,
        custom_image_uri=None,
        **kwargs,
    ):
        image_uri = custom_image_uri
        if image_uri is None:
            image_uri = image_uris.retrieve(
                "autogluon",
                region=region,
                version=framework_version,
                py_version=py_version,
                image_scope="inference",
                instance_type=instance_type,
            )
        # setting PYTHONUNBUFFERED to disable output buffering for endpoints logging
        super().__init__(
            model_data=model_data,
            role=role,
            entry_point=entry_point,
            image_uri=image_uri,
            env={"PYTHONUNBUFFERED": "1"},
            **kwargs,
        )

    def transformer(
        self,
        instance_count,
        instance_type,
        strategy="MultiRecord",
        # Maximum size of the payload in a single HTTP request to the container in MB. Will split into multiple batches if a request is more than max_payload
        max_payload=6,
        max_concurrent_transforms=1,  # The maximum number of HTTP requests to be made to each individual transform container at one time.
        accept="application/json",
        assemble_with="Line",
        **kwargs,
    ):
        return super().transformer(
            instance_count=instance_count,
            instance_type=instance_type,
            strategy=strategy,
            max_payload=max_payload,
            max_concurrent_transforms=max_concurrent_transforms,
            accept=accept,
            assemble_with=assemble_with,
            **kwargs,
        )


class AutoGluonRepackInferenceModel(AutoGluonSagemakerInferenceModel):
    """
    Custom implementation to force repack of inference code into model artifacts
    """

    def prepare_container_def(
        self,
        instance_type=None,
        accelerator_type=None,
        serverless_inference_config=None,
    ):  # pylint: disable=unused-argument
        deploy_key_prefix = fw_utils.model_code_key_prefix(
            self.key_prefix, self.name, self.image_uri
        )
        deploy_env = copy.deepcopy(self.env)
        self._upload_code(deploy_key_prefix, repack=True)
        deploy_env.update(self._script_mode_env_vars())
        return sagemaker.container_def(
            self.image_uri,
            self.repacked_model_data or self.model_data,
            deploy_env,
            image_config=self.image_config,
        )


class AutoGluonNonRepackInferenceModel(AutoGluonSagemakerInferenceModel):
    """
    Custom implementation to force no repack of inference code into model artifacts.
    This requires inference code already present in the trained artifacts, which is created during CloudPredictor training.
    """

    def prepare_container_def(
        self,
        instance_type=None,
        accelerator_type=None,
        serverless_inference_config=None,
    ):  # pylint: disable=unused-argument
        deploy_env = copy.deepcopy(self.env)
        deploy_env.update(self._script_mode_env_vars())
        deploy_env[SCRIPT_PARAM_NAME.upper()] = os.path.basename(
            deploy_env[SCRIPT_PARAM_NAME.upper()]
        )
        deploy_env[DIR_PARAM_NAME.upper()] = "/opt/ml/model/code"

        return sagemaker.container_def(
            self.image_uri,
            self.model_data,
            deploy_env,
            image_config=self.image_config,
        )


# Predictor documentation: https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html
class AutoGluonRealtimePredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            serializer=ParquetSerializer(),
            deserializer=PandasDeserializer(),
            **kwargs,
        )


# Predictor documentation: https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html
# SageMaker can only take in csv format for batch transformation because files need to be easily splitable to be batch processed.
class AutoGluonBatchPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, serializer=CSVSerializer(), **kwargs)
