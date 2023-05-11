from typing import Type
from sagemaker import image_uris, model_uris, instance_types, environment_variables
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.session import Session
from sagemaker.serializers import BaseSerializer, JSONSerializer
from sagemaker.deserializers import BaseDeserializer, JSONDeserializer
from sagemaker.utils import name_from_base


def deploy_jumpstart_model(
    model_id: str,
    sagemaker_session: Session,
    model_version: str = "*",
    serializer_class: Type[BaseSerializer] = JSONSerializer,
    deserializer_class: Type[BaseDeserializer] = JSONDeserializer,
) -> Predictor:
    """Create and deploy a SageMaker JumpStart model."""
    aws_role = sagemaker_session.get_caller_identity_arn()
    region = sagemaker_session.boto_region_name

    endpoint_name = name_from_base(f"jumpstart-bm-{model_id.replace('huggingface', 'hf')}")

    instance_type = instance_types.retrieve_default(
        region=region, model_id=model_id, model_version=model_version, scope="inference"
    )

    image_uri = image_uris.retrieve(
        region=region,
        framework=None,
        image_scope="inference",
        model_id=model_id,
        model_version=model_version,
        instance_type=instance_type,
    )

    model_uri = model_uris.retrieve(model_id=model_id, model_version=model_version, model_scope="inference")

    env_vars = environment_variables.retrieve_default(
        model_id=model_id,
        model_version=model_version,
    )

    model = Model(
        image_uri=image_uri,
        model_data=model_uri,
        role=aws_role,
        predictor_cls=Predictor,
        name=endpoint_name,
        env=env_vars,
        sagemaker_session=sagemaker_session,
    )

    serializer = serializer_class()
    deserializer = deserializer_class()

    print(f"(Model {model_id}): Deploying endpoint {endpoint_name} ...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        predictor_cls=Predictor,
        endpoint_name=endpoint_name,
        serializer=serializer,
        deserializer=deserializer,
    )
    print(f"\n(Model {model_id}): Successfully deployed endpoint {endpoint_name} ...")

    return predictor
