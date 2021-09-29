from sagemaker.estimator import Framework
from sagemaker.predictor import Predictor
from sagemaker.mxnet import MXNetModel
from sagemaker.mxnet.model import MXNetPredictor
from sagemaker import utils
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import StringDeserializer


ACCOUNT = 763104351884
ECR_TRAINING_REPO = "autogluon-training"
ECR_INFERENCE_REPO = "autogluon-inference"
TRAINING_IMAGE_CPU = "cpu-py37-ubuntu18.04"
TRAINING_IMAGE_GPU = "gpu-py37-cu102-ubuntu18.04"
INFERENCE_IMAGE_CPU = "cpu-py37-ubuntu16.04"


class AutoGluonTraining(Framework):
    def __init__(
        self,
        entry_point,
        region,
        framework_version,
        image_type="cpu",
        source_dir=None,
        hyperparameters=None,
        **kwargs,
    ):
        image = TRAINING_IMAGE_GPU if image_type == "gpu" else TRAINING_IMAGE_CPU
        image = f"{framework_version}-{image}"
        image_uri = f"{ACCOUNT}.dkr.ecr.{region}.amazonaws.com/{ECR_TRAINING_REPO}:{image}"
        super().__init__(entry_point, source_dir, hyperparameters, image_uri=image_uri, **kwargs)

    def _configure_distribution(self, distributions):
        return

    def create_model(
        self,
        model_server_workers=None,
        role=None,
        vpc_config_override=None,
        entry_point=None,
        source_dir=None,
        dependencies=None,
        image_name=None,
        **kwargs,
    ):
        return None


class AutoGluonTabularPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, serializer=CSVSerializer(), deserializer=StringDeserializer(), **kwargs
        )


class AutoGluonInferenceModel(MXNetModel):
    def __init__(self, model_data, role, entry_point, region, framework_version, **kwargs):
        image = f"{framework_version}-{INFERENCE_IMAGE_CPU}"
        image_uri = f"{ACCOUNT}.dkr.ecr.{region}.amazonaws.com/{ECR_INFERENCE_REPO}:{image}"
        super().__init__(
            model_data, role, entry_point, image_uri=image_uri, framework_version="1.8.0", **kwargs
        )
