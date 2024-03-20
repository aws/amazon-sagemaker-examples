from concurrent import futures
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Type
from typing import Dict
from typing import List

import pandas as pd
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.predictor import Predictor
from sagemaker.predictor import retrieve_default
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.session import Session
from sagemaker.utils import name_from_base
from sagemaker import image_uris
from sagemaker.model import Model

from benchmarking.clients import PricingClient
from benchmarking.clients import SageMakerClient
from benchmarking.concurrency_probe import num_invocation_scaler
from benchmarking.concurrency_probe import ConcurrentProbeIteratorBase
from benchmarking.concurrency_probe import ConcurrentProbeExponentialScalingIterator
from benchmarking.constants import MAX_CONCURRENT_BENCHMARKS, SAVE_METRICS_FILE_PATH
from benchmarking.constants import MAX_CONCURRENT_INVOCATIONS_PER_MODEL
from benchmarking.constants import MAX_TOTAL_RETRY_TIME_SECONDS
from benchmarking.constants import NUM_INVOCATIONS
from benchmarking.constants import RETRY_WAIT_TIME_SECONDS
from benchmarking.constants import SM_SESSION
from benchmarking.load_test import LoadTester
from benchmarking.logging import logging_prefix
from benchmarking.custom_predictor import CustomPredictor


class Benchmarker:
    def __init__(
        self,
        payloads: Dict[str, Dict[str, Any]],
        max_concurrent_benchmarks: int = MAX_CONCURRENT_BENCHMARKS,
        sagemaker_session: Session = SM_SESSION,
        num_invocations: int = NUM_INVOCATIONS,
        max_workers: int = MAX_CONCURRENT_INVOCATIONS_PER_MODEL,
        retry_wait_time: float = RETRY_WAIT_TIME_SECONDS,
        max_total_retry_time: float = MAX_TOTAL_RETRY_TIME_SECONDS,
        run_latency_load_test: bool = False,
        run_throughput_load_test: bool = False,
        run_concurrency_probe: bool = False,
        concurrency_probe_num_invocation_hook: Optional[Callable[[int], int]] = None,
        concurrency_probe_concurrent_request_iterator_cls: Optional[Type[ConcurrentProbeIteratorBase]] = None,
        clean_up: bool = False,
        attempt_retrieve_predictor: bool = True,
        saved_metrics_path: Path = SAVE_METRICS_FILE_PATH,
    ):
        self.payloads = payloads
        self.max_concurrent_benchmarks = max_concurrent_benchmarks
        self.sagemaker_session = sagemaker_session
        self.num_invocations = num_invocations
        self.max_workers = max_workers
        self.retry_wait_time = retry_wait_time
        self.max_total_retry_time = max_total_retry_time
        self.run_latency_load_test = run_latency_load_test
        self.run_throughput_load_test = run_throughput_load_test
        self.run_concurrency_probe = run_concurrency_probe

        if concurrency_probe_num_invocation_hook is None:
            self.concurrency_probe_num_invocation_hook = num_invocation_scaler
        else:
            self.concurrency_probe_num_invocation_hook = concurrency_probe_num_invocation_hook

        if concurrency_probe_concurrent_request_iterator_cls is None:
            self.concurrency_probe_concurrent_request_iterator_cls = ConcurrentProbeExponentialScalingIterator
        else:
            self.concurrency_probe_concurrent_request_iterator_cls = concurrency_probe_concurrent_request_iterator_cls

        self.clean_up = clean_up
        self._pricing_client = PricingClient()
        self._sagemaker_client = SageMakerClient()
        self.model_id_to_endpoint_name: Dict[str, str] = {}
        if attempt_retrieve_predictor:
            self.model_id_to_endpoint_name = self.load_metrics_json(saved_metrics_path).get("endpoints", {})

    def _run_benchmarking_tests(
        self,
        predictor: CustomPredictor,
        payload: Dict[str, Any],
        model_id: str,
        payload_name: str,
        tokenizer_model_id: str,
        huggingface_hub_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        metrics_latency: Dict[str, Any] = {}
        metrics_throughput: Dict[str, Any] = {}
        metrics_concurrency: Dict[str, Any] = {}

        if predictor.predictor is not None:
            endpoint_description = self._sagemaker_client.describe_endpoint(predictor.endpoint_name)
            endpoint_config_name = endpoint_description["EndpointConfigName"]
            endpoint_config_description = self._sagemaker_client.describe_endpoint_config(endpoint_config_name)
            model_description = self._sagemaker_client.describe_model(predictor.endpoint_name)
            production_variant = endpoint_config_description["ProductionVariants"][0]
            # primary_container = model_description["PrimaryContainer"]
            instance_type = production_variant["InstanceType"]
            price_per_instance = self._pricing_client.get_price_per_unit(instance_type, SM_SESSION._region_name)
            price_per_endpoint = production_variant["InitialInstanceCount"] * price_per_instance
            metrics_pricing = {
                "PricePerInstance": price_per_instance,
                "PricePerEndpoint": price_per_endpoint,
            }
            creation_time: datetime = endpoint_description["CreationTime"]
            last_modified_time: datetime = endpoint_description["LastModifiedTime"]
            metrics_time = {
                "CreationTime": creation_time.isoformat(),
                "LastModifiedTime": last_modified_time.isoformat(),
                "DeploymentTime": (last_modified_time - creation_time).seconds,
            }
        else:
            if hasattr(predictor, "instance_type") and predictor.instance_type is not None:
                instance_type = predictor.instance_type
            else:
                logging.info(
                    f"{logging_prefix(model_id)} No instance type provided. Using the default ml.g5.2xlarge for pricing calculations."
                )
                instance_type = "ml.g5.2xlarge"
            if hasattr(predictor, "instance_count") and predictor.instance_count is not None:
                initial_instance_count = predictor.instance_count
            else:
                logging.info(
                    f"{logging_prefix(model_id)} No initial_instance_count provided. Using the default count 1."
                )
                initial_instance_count = 1
            price_per_instance = self._pricing_client.get_price_per_unit(instance_type, SM_SESSION._region_name)
            price_per_endpoint = initial_instance_count * price_per_instance
            metrics_pricing = {
                "PricePerInstance": price_per_instance,
                "PricePerEndpoint": price_per_endpoint,
            }
            metrics_time = {
                "CreationTime": 0,
                "LastModifiedTime": 0,
                "DeploymentTime": 0,
            }
            production_variant = {
                "VariantName": "AllTraffic",
                "ModelName": predictor.endpoint_name,
                "InitialInstanceCount": initial_instance_count,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
                "ModelDataDownloadTimeoutInSeconds": 3600,
                "ContainerStartupHealthCheckTimeoutInSeconds": 3600,
            }

        tester = LoadTester(
            predictor,
            payload,
            model_id,
            payload_name,
            tokenizer_model_id,
            huggingface_hub_token,
            price_per_endpoint,
        )

        if self.run_latency_load_test:
            metrics_latency = tester.run_latency_load_test(self.num_invocations)
        if self.run_throughput_load_test:
            metrics_throughput = tester.run_throughput_load_test(self.num_invocations, self.max_workers)
        if self.run_concurrency_probe:
            concurrency_probe_results = tester.run_concurrency_probe(
                iterator_cls=self.concurrency_probe_concurrent_request_iterator_cls,
                num_invocation_hook=self.concurrency_probe_num_invocation_hook,
            )
            metrics_concurrency = {"ConcurrencyProbe": concurrency_probe_results}

        return {
            **metrics_latency,
            **metrics_throughput,
            **metrics_concurrency,
            **metrics_pricing,
            **metrics_time,
            "ProductionVariant": production_variant,
        }

    def run_single_predictor(
        self,
        model_id: str,
        predictor: CustomPredictor,
        tokenizer_model_id: Optional[str] = None,
        huggingface_hub_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run benchmarker given a Predictor for an in-service model endpoint."""
        metrics = []
        try:
            for payload_name, payload in self.payloads.items():
                metrics_payload = self._run_benchmarking_tests(
                    predictor,
                    payload,
                    model_id,
                    payload_name,
                    tokenizer_model_id,
                    huggingface_hub_token,
                )
                metrics.append(metrics_payload)
        finally:
            if self.clean_up is True:
                self.clean_up_predictor(model_id, predictor)
            else:
                logging.info(f"{logging_prefix(model_id)} Skipping cleaning up resources ...")

        return metrics

    def run_single_model(
        self, model_id: str, model_args: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], CustomPredictor]:
        """Run benchmarker for a single model.

        If an `endpoint_name` is provided either as a key in `model_args` or saved in benchmarking metrics file from
        a previous invocation of this benchmarker, then a predictor is attempted to be attached to this endpoint. If
        an `endpoint_name` is not provided, then the model is deployed prior to benchmarking run.
        """
        endpoint_name = model_args.get("endpoint_name") or self.model_id_to_endpoint_name.get(model_id)
        endpoint_url = model_args.get("endpoint_url")
        instance_type = model_args.get("instance_type")
        if endpoint_url is not None:
            predictor = CustomPredictor(endpoint_url=endpoint_url, instance_type=instance_type)
        elif endpoint_name is not None:
            try:
                predictor = self.retrieve_predictor_from_endpoint(endpoint_name, model_args)
                predictor = CustomPredictor(predictor=predictor)
                logging.info(f"{logging_prefix(model_id)} Predictor successfully retrieved from endpoint name")
            except Exception as e:
                logging.warning(f"{logging_prefix(model_id)} Failed to retrieve predictor, re-deploying model: {e}")
                predictor = self.deploy_model(model_id, model_args)
                predictor = CustomPredictor(predictor=predictor)
        else:
            predictor = self.deploy_model(model_id, model_args)
            predictor = CustomPredictor(predictor=predictor)

        self.model_id_to_endpoint_name[model_id] = predictor.endpoint_name
        metrics = self.run_single_predictor(
            model_id,
            predictor,
            model_args["huggingface_model_id"],
            model_args.get("huggingface_hub_token"),
        )
        return metrics, predictor

    def retrieve_predictor_from_endpoint(
        self, endpoint_name: str, model_args: Optional[Dict[str, Any]] = None
    ) -> Predictor:
        """Obtain a predictor from an already deployed endpoint."""
        if model_args is not None:
            jumpstart_model_args: Dict[str, Any] = model_args.get("jumpstart_model_args")
            if jumpstart_model_args:
                return retrieve_default(
                    endpoint_name=endpoint_name,
                    model_id=jumpstart_model_args["model_id"],
                    model_version=jumpstart_model_args.get("model_version", "*"),
                    sagemaker_session=self.sagemaker_session,
                )

        return Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=self.sagemaker_session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )

    def deploy_model(self, model_id: str, model_args: Dict[str, Any]) -> Predictor:
        """Deploy a model with configuration defined by model_args.

        Two model deployment methods are supported:
        - Use JumpStartModel object with kwargs defined in `jumpstart_model_specs` key.
        - Use Model object with `image_uri_args`, `model_args`, and `deploy_args` kwards defined in `model_specs` key.

        Raises:
            ValueError: if neither `jumpstart_model_specs` or `model_specs` keys are present in model_args.
        """
        jumpstart_model_specs: Optional[Dict[str, Any]] = model_args.get("jumpstart_model_specs")
        model_specs: Optional[Dict[str, Any]] = model_args.get("model_specs")
        endpoint_name = name_from_base(f"bm-{model_id.replace('huggingface', 'hf')}")
        logging.info(f"{logging_prefix(model_id)} Deploying endpoint {endpoint_name} ...")
        if jumpstart_model_specs:
            model = JumpStartModel(
                sagemaker_session=self.sagemaker_session,
                **jumpstart_model_specs["model_args"],
            )
            return model.deploy(
                endpoint_name=endpoint_name,
                **jumpstart_model_specs.get("deploy_args", {}),
            )
        elif model_specs:
            image_uri = image_uris.retrieve(region=SM_SESSION._region_name, **model_specs["image_uri_args"])
            model = Model(
                image_uri=image_uri,
                role=SM_SESSION.get_caller_identity_arn(),
                predictor_cls=Predictor,
                name=endpoint_name,
                **model_specs["model_args"],
            )
            return model.deploy(
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer(),
                endpoint_name=endpoint_name,
                **model_specs["deploy_args"],
            )
        else:
            raise ValueError(f"{logging_prefix(model_id)} No model arguments discovered for deployment.")

    def run_multiple_models(
        self,
        models: Dict[str, Dict[str, Any]],
        save_file_path: Path = SAVE_METRICS_FILE_PATH,
    ) -> Dict[str, Any]:
        """Concurrently call run_single_model for all models and aggregate benchmarking output."""
        metrics = []
        errors = {}
        endpoints: Dict[str, str] = {}

        with futures.ThreadPoolExecutor(max_workers=self.max_concurrent_benchmarks) as executor:
            future_to_model_id = {
                executor.submit(self.run_single_model, model_id, args): model_id for model_id, args in models.items()
            }
            for future in futures.as_completed(future_to_model_id):
                model_id = future_to_model_id[future]
                try:
                    metrics_model_id, predictor = future.result()
                    endpoints[model_id] = predictor.endpoint_name
                    metrics.extend(metrics_model_id)
                except Exception as e:
                    errors[model_id] = e
                    logging.error(f"{logging_prefix(model_id)} Benchmarking failed: {e}")

        output = {
            "models": models,
            "payloads": self.payloads,
            "endpoints": endpoints,
            "metrics": metrics,
        }

        with open(save_file_path, "w") as file:
            json.dump(output, file, indent=4, ensure_ascii=False)

        return output

    @classmethod
    def load_metrics_pandas(cls, save_file_path: Path = SAVE_METRICS_FILE_PATH) -> pd.DataFrame:
        """Create a pandas DataFrame from the saved JSON metrics file."""
        metrics = cls.load_metrics_json(save_file_path)
        return pd.json_normalize(
            data=metrics,
            record_path=["metrics", "ConcurrencyProbe"],
            meta=[
                ["metrics", "ProductionVariant", "InstanceType"],
                ["metrics", "ProductionVariant", "InitialInstanceCount"],
                ["metrics", "PrimaryContainer", "Image"],
                ["metrics", "PricePerEndpoint"],
                ["metrics", "PricePerInstance"],
                ["metrics", "DeploymentTime"],
            ],
            errors="ignore",
        )

    @staticmethod
    def create_concurrency_probe_pivot_table(
        df: pd.DataFrame,
        value_format_dict: Optional[Dict[str, Callable]] = None,
        value_name_dict: Optional[Dict[str, str]] = None,
        fillna_str: str = "--",
    ) -> pd.DataFrame:
        """Pivot concurrency probe pandas DataFrame to show specified values across models and concurrent requests."""
        if value_format_dict is None:
            value_format_dict = {
                "TokenThroughput": "{:.2f}".format,
                "LatencyPerToken.p90": int,
                "CostToGenerate1MTokens": "${:,.2f}".format,
            }
        if value_name_dict is None:
            value_name_dict = {
                "LatencyPerToken.p90": "p90 latency (ms/token)",
                "TokenThroughput": "throughput (tokens/s)",
                "CostToGenerate1MTokens": "cost to generate 1M tokens ($)",
            }

        df_copy = df.copy()

        index_cols = [
            "ModelID",
            "metrics.ProductionVariant.InstanceType",
            "PayloadName",
        ]
        columns_cols = ["ConcurrentRequests"]
        value_cols = value_format_dict.keys()

        for value_name, mapping_function in value_format_dict.items():
            df_copy[value_name] = df_copy[value_name].map(mapping_function)

        df_pivot = df_copy.pivot(index=index_cols, columns=columns_cols, values=value_cols).fillna(fillna_str)
        df_pivot = df_pivot.rename(columns=value_name_dict)
        df_pivot.index = df_pivot.index.rename(["model ID", "instance type", "payload"])
        df_pivot.columns = df_pivot.columns.rename([None, "concurrent requests"])
        return df_pivot

    def clean_up_resources(self) -> None:
        """Delete model and endpoint for all endpoints attached to this benchmarker."""
        for model_id, endpoint_name in self.model_id_to_endpoint_name.items():
            predictor = self.retrieve_predictor_from_endpoint(endpoint_name)
            self.clean_up_predictor(model_id, predictor)

    @classmethod
    def clean_up_predictor(cls, model_id: str, predictor: CustomPredictor) -> None:
        """Delete model and endpoint for a single predictor."""
        logging.info(f"{logging_prefix(model_id)} Cleaning up resources ...")
        predictor.delete_model()
        predictor.delete_endpoint()

    @staticmethod
    def load_metrics_json(
        save_file_path: Path = SAVE_METRICS_FILE_PATH,
    ) -> Dict[str, str]:
        """Attempt to load metrics from a previous benchmarking run."""
        try:
            with open(save_file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to extract endpoint names from saved benchmarking file: {e}")
            return {}

        return data
