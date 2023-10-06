from concurrent import futures
import json
from pathlib import Path
from typing import Any, Optional, Tuple
from typing import Dict
from typing import List

import boto3
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

from benchmarking.constants import MAX_CONCURRENT_BENCHMARKS, SAVE_METRICS_FILE_PATH
from benchmarking.constants import MAX_CONCURRENT_INVOCATIONS_PER_MODEL
from benchmarking.constants import MAX_TOTAL_RETRY_TIME_SECONDS
from benchmarking.constants import NUM_INVOCATIONS
from benchmarking.constants import RETRY_WAIT_TIME_SECONDS
from benchmarking.constants import SM_SESSION
from benchmarking.load_test import LoadTester
from benchmarking.load_test import logging_prefix
from benchmarking.pricing_client import PricingClient


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
        clean_up: bool = False,
        attempt_retrieve_predictor: bool = True,
        saved_metrics_path: Path = SAVE_METRICS_FILE_PATH
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
        self.clean_up = clean_up
        self._pricing_client = PricingClient()
        self.endpoints_from_file: Dict[str, str] = {}
        if attempt_retrieve_predictor:
            self.endpoints_from_file = self.load_metrics_json(saved_metrics_path).get("endpoints", {})

    def _run_benchmarking_tests(self, tester: LoadTester) -> Dict[str, Any]:
        metrics_latency: Dict[str, Any] = {}
        metrics_throughput: Dict[str, Any] = {}
        metrics_concurrency: Dict[str, Any] = {}

        if self.run_latency_load_test:
            metrics_latency = tester.run_latency_load_test(self.num_invocations)
        if self.run_throughput_load_test:
            metrics_throughput = tester.run_throughput_load_test(self.num_invocations, self.max_workers)
        if self.run_concurrency_probe:
            metrics_concurrency = {"ConcurrencyProbe": tester.run_concurrency_probe()}

        endpoint_config = describe_endpoint_config(tester.predictor.endpoint_name)
        instance_type = endpoint_config["ProductionVariants"][0]["InstanceType"]
        price_per_unit = self._pricing_client.get_price_per_unit(instance_type, SM_SESSION._region_name)

        return {
            **metrics_latency,
            **metrics_throughput,
            **metrics_concurrency,
            "InstanceType": instance_type,
            "PricePerUnit": price_per_unit
        }
    
    def run_single_predictor(
        self, model_id: str, predictor: Predictor, tokenizer_model_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Run benchmarker given a Predictor for an in-service model endpoint."""
        metrics = []
        try:
            for payload_name, payload in self.payloads.items():
                tester = LoadTester(predictor, payload, model_id, payload_name, tokenizer_model_id)
                metrics_payload = self._run_benchmarking_tests(tester)
                metrics.append(metrics_payload)
        finally:
            if self.clean_up is True:
                print(f"{logging_prefix(model_id)} Cleaning up resources ...")
                predictor.delete_model()
                predictor.delete_endpoint()
            else:
                print(f"{logging_prefix(model_id)} Skipping cleaning up resources ...")

        return metrics

    def run_single_model(self, model_id: str, model_args: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Predictor]:
        """Run benchmarker for a single model.
        
        If an `endpoint_name` is provided either as a key in `model_args` or saved in benchmarking metrics file from
        a previous invocation of this benchmarker, then a predictor is attempted to be attached to this endpoint. If
        an `endpoint_name` is not provided, then the model is deployed prior to benchmarking run.
        """
        endpoint_name = model_args.get("endpoint_name") or self.endpoints_from_file.get(model_id)
        if endpoint_name is not None:
            try:
                predictor = self.retrieve_predictor_from_endpoint(endpoint_name, model_args)
                print(f"{logging_prefix(model_id)} Predictor successfully retrieved from endpoint name")
            except Exception as e:
                print(f"{logging_prefix(model_id)} Failed to retrieve predictor, re-deploying model: {e}")
                predictor = self.deploy_model(model_id, model_args)
        else:
            pass
            predictor = self.deploy_model(model_id, model_args)

        metrics = self.run_single_predictor(model_id, predictor, model_args["huggingface_model_id"])
        return metrics, predictor
    
    def retrieve_predictor_from_endpoint(self, endpoint_name: str, model_args: Dict[str, Any]) -> Predictor:
        """Obtain a predictor from an already deployed endpoint."""
        jumpstart_model_args = model_args.get("jumpstart_model_args", None)
        if jumpstart_model_args:
            return retrieve_default(
                endpoint_name=endpoint_name,
                model_id=jumpstart_model_args["model_id"],
                model_version="*",
                sagemaker_session=self.sagemaker_session
            )
        else:
            return Predictor(
                endpoint_name=endpoint_name,
                sagemaker_session=self.sagemaker_session,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer(),
            )
    
    def deploy_model(self, model_id: str, model_args: Dict[str, Any]) -> Predictor:
        """Deploy a model with configuration defined by model_args.
        
        Two model deployment methods are supported:
        - Use JumpStartModel object with kwargs defined in `jumpstart_model_args` key.
        - Use Model object with `image_uri_args`, `model_args`, and `deploy_args` kwards defined in `model_specs` key.

        Raises:
            ValueError: if neither `jumpstart_model_args` or `model_specs` keys are present in model_args.
        """
        jumpstart_model_args = model_args.get("jumpstart_model_args", None)
        model_specs = model_args.get("model_specs", None)
        endpoint_name = name_from_base(f"bm-{model_id.replace('huggingface', 'hf')}")
        print(f"{logging_prefix(model_id)} Deploying endpoint {endpoint_name} ...")
        if jumpstart_model_args:
            model = JumpStartModel(sagemaker_session=self.sagemaker_session, **jumpstart_model_args)
            return model.deploy(endpoint_name=endpoint_name)
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

    def run_multiple_model_ids(
        self, models: Dict[str, Dict[str, Any]], save_file_path: Path = SAVE_METRICS_FILE_PATH
    ) -> Dict[str, Any]:
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
                    print(f"{logging_prefix(model_id)} Benchmarking failed: {e}")

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
        metrics = cls.load_metrics_json(save_file_path)
        return pd.json_normalize(metrics, ["metrics", "ConcurrencyProbe"], [["metrics", "InstanceType"], ["metrics", "PricePerUnit"]])
    
    @staticmethod
    def create_concurrency_probe_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()

        df_copy["CostToGenerate1MTokens"] = df_copy["TimeToGenerate1MTokens"] * df_copy["metrics.PricePerUnit"].astype(float)

        index_cols = ["ModelID", "metrics.InstanceType", "PayloadName"]
        columns_cols = ["ConcurrentRequests"]
        value_cols = ["TokenThroughput", "LatencyPerToken.p50", "CostToGenerate1MTokens"]

        df_copy = df_copy.astype({"TokenThroughput": "int", "LatencyPerToken.p50": "int"})
        df_copy["CostToGenerate1MTokens"] = df_copy["CostToGenerate1MTokens"].map("${:,.2f}".format)
        df_copy = df_copy.astype({"TokenThroughput": "str", "LatencyPerToken.p50": "str", "CostToGenerate1MTokens": "str"})

        df_pivot = df_copy.pivot(index=index_cols, columns=columns_cols, values=value_cols).fillna("--")
        df_pivot = df_pivot.rename(
            columns={
                "LatencyPerToken.p50": "p50 latency (ms/token)",
                "TokenThroughput": "throughput (tokens/s)",
                "CostToGenerate1MTokens": "cost to generate 1M tokens ($)",
            }
        )
        df_pivot.index = df_pivot.index.rename(["model ID", "instance type", "payload"])
        df_pivot.columns = df_pivot.columns.rename([None, "concurrent users"])
        return df_pivot

    @staticmethod
    def load_metrics_json(save_file_path: Path = SAVE_METRICS_FILE_PATH) -> Dict[str, str]:
        try:
            with open(save_file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to extract endpoint names from saved benchmarking file: {e}")
            return {}
        
        return data
    

def describe_endpoint_config(endpoint_name: str) -> Dict[str, Any]:
    sagemaker = boto3.client("sagemaker")
    return sagemaker.describe_endpoint_config(EndpointConfigName=endpoint_name)
