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
"""This module stores notebook utils related to SageMaker JumpStart."""
from __future__ import absolute_import
import copy

from functools import cmp_to_key
from typing import Any, Generator, List, Optional, Tuple, Union, Set, Dict
from packaging.version import Version
from sagemaker.jumpstart import accessors
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.filters import (
    SPECIAL_SUPPORTED_FILTER_KEYS,
    BooleanValues,
    Identity,
    SpecialSupportedFilterKeys,
)
from sagemaker.jumpstart.filters import Constant, ModelFilter, Operator, evaluate_filter_expression
from sagemaker.jumpstart.utils import get_sagemaker_version


def _compare_model_version_tuples(  # pylint: disable=too-many-return-statements
    model_version_1: Optional[Tuple[str, str]] = None,
    model_version_2: Optional[Tuple[str, str]] = None,
) -> int:
    """Performs comparison of sdk specs paths, in order to sort them.

    Args:
        model_version_1 (Tuple[str, str]): The first model ID and version tuple to compare.
        model_version_2 (Tuple[str, str]): The second model ID and version tuple to compare.
    """
    if model_version_1 is None or model_version_2 is None:
        if model_version_2 is not None:
            return -1
        if model_version_1 is not None:
            return 1
        return 0

    model_id_1, version_1 = model_version_1

    model_id_2, version_2 = model_version_2

    if model_id_1 < model_id_2:
        return -1

    if model_id_2 < model_id_1:
        return 1

    if Version(version_1) < Version(version_2):
        return 1

    if Version(version_2) < Version(version_1):
        return -1

    return 0


def _model_filter_in_operator_generator(filter_operator: Operator) -> Generator:
    """Generator for model filters in an operator."""
    for operator in filter_operator:
        if isinstance(operator.unresolved_value, ModelFilter):
            yield operator


def _put_resolved_booleans_into_filter(
    filter_operator: Operator, model_filters_to_resolved_values: Dict[ModelFilter, BooleanValues]
) -> None:
    """Iterate over the operators in the filter, assign resolved value if found in second arg.

    If not found, assigns ``UNKNOWN``.
    """
    for operator in _model_filter_in_operator_generator(filter_operator):
        model_filter = operator.unresolved_value
        operator.resolved_value = model_filters_to_resolved_values.get(
            model_filter, BooleanValues.UNKNOWN
        )


def _populate_model_filters_to_resolved_values(
    manifest_specs_cached_values: Dict[str, Any],
    model_filters_to_resolved_values: Dict[ModelFilter, BooleanValues],
    model_filters: Operator,
) -> None:
    """Iterate over the model filters, if the filter key has a cached value, evaluate the filter.

    The resolved filter values are placed in ``model_filters_to_resolved_values``.
    """
    for model_filter in model_filters:
        if model_filter.key in manifest_specs_cached_values:
            cached_model_value = manifest_specs_cached_values[model_filter.key]
            evaluated_expression: BooleanValues = evaluate_filter_expression(
                model_filter, cached_model_value
            )
            model_filters_to_resolved_values[model_filter] = evaluated_expression


def extract_framework_task_model(model_id: str) -> Tuple[str, str, str]:
    """Parse the model ID, return a tuple framework, task, rest-of-id.

    Args:
        model_id (str): The model ID for which to extract the framework/task/model.

    Raises:
        ValueError: If the model ID cannot be parsed into at least 3 components seperated by
            "-" character.
    """
    _id_parts = model_id.split("-")

    if len(_id_parts) < 3:
        raise ValueError(f"incorrect model ID: {model_id}.")

    framework = _id_parts[0]
    task = _id_parts[1]
    name = "-".join(_id_parts[2:])

    return framework, task, name


def list_jumpstart_tasks(  # pylint: disable=redefined-builtin
    filter: Union[Operator, str] = Constant(BooleanValues.TRUE),
    region: str = JUMPSTART_DEFAULT_REGION_NAME,
) -> List[str]:
    """List tasks for JumpStart, and optionally apply filters to result.

    Args:
        filter (Union[Operator, str]): Optional. The filter to apply to list tasks. This can be
            either an ``Operator`` type filter (e.g. ``And("task == ic", "framework == pytorch")``),
            or simply a string filter which will get serialized into an Identity filter.
            (e.g. ``"task == ic"``). If this argument is not supplied, all tasks will be listed.
            (Default: Constant(BooleanValues.TRUE)).
        region (str): Optional. The AWS region from which to retrieve JumpStart metadata regarding
            models. (Default: JUMPSTART_DEFAULT_REGION_NAME).
    """

    tasks: Set[str] = set()
    for model_id, _ in _generate_jumpstart_model_versions(filter=filter, region=region):
        _, task, _ = extract_framework_task_model(model_id)
        tasks.add(task)
    return sorted(list(tasks))


def list_jumpstart_frameworks(  # pylint: disable=redefined-builtin
    filter: Union[Operator, str] = Constant(BooleanValues.TRUE),
    region: str = JUMPSTART_DEFAULT_REGION_NAME,
) -> List[str]:
    """List frameworks for JumpStart, and optionally apply filters to result.

    Args:
        filter (Union[Operator, str]): Optional. The filter to apply to list frameworks. This can be
            either an ``Operator`` type filter (e.g. ``And("task == ic", "framework == pytorch")``),
            or simply a string filter which will get serialized into an Identity filter.
            (eg. ``"task == ic"``). If this argument is not supplied, all frameworks will be listed.
            (Default: Constant(BooleanValues.TRUE)).
        region (str): Optional. The AWS region from which to retrieve JumpStart metadata regarding
            models. (Default: JUMPSTART_DEFAULT_REGION_NAME).
    """

    frameworks: Set[str] = set()
    for model_id, _ in _generate_jumpstart_model_versions(filter=filter, region=region):
        framework, _, _ = extract_framework_task_model(model_id)
        frameworks.add(framework)
    return sorted(list(frameworks))


def list_jumpstart_scripts(  # pylint: disable=redefined-builtin
    filter: Union[Operator, str] = Constant(BooleanValues.TRUE),
    region: str = JUMPSTART_DEFAULT_REGION_NAME,
) -> List[str]:
    """List scripts for JumpStart, and optionally apply filters to result.

    Args:
        filter (Union[Operator, str]): Optional. The filter to apply to list scripts. This can be
            either an ``Operator`` type filter (e.g. ``And("task == ic", "framework == pytorch")``),
            or simply a string filter which will get serialized into an Identity filter.
            (e.g. ``"task == ic"``). If this argument is not supplied, all scripts will be listed.
            (Default: Constant(BooleanValues.TRUE)).
        region (str): Optional. The AWS region from which to retrieve JumpStart metadata regarding
            models. (Default: JUMPSTART_DEFAULT_REGION_NAME).
    """
    if (isinstance(filter, Constant) and filter.resolved_value == BooleanValues.TRUE) or (
        isinstance(filter, str) and filter.lower() == BooleanValues.TRUE.lower()
    ):
        return sorted([e.value for e in JumpStartScriptScope])

    scripts: Set[str] = set()
    for model_id, version in _generate_jumpstart_model_versions(filter=filter, region=region):
        scripts.add(JumpStartScriptScope.INFERENCE)
        model_specs = accessors.JumpStartModelsAccessor.get_model_specs(
            region=region,
            model_id=model_id,
            version=version,
        )
        if model_specs.training_supported:
            scripts.add(JumpStartScriptScope.TRAINING)

        if scripts == {e.value for e in JumpStartScriptScope}:
            break
    return sorted(list(scripts))


def list_jumpstart_models(  # pylint: disable=redefined-builtin
    filter: Union[Operator, str] = Constant(BooleanValues.TRUE),
    region: str = JUMPSTART_DEFAULT_REGION_NAME,
    list_incomplete_models: bool = False,
    list_old_models: bool = False,
    list_versions: bool = False,
) -> List[Union[Tuple[str], Tuple[str, str]]]:
    """List models for JumpStart, and optionally apply filters to result.

    Args:
        filter (Union[Operator, str]): Optional. The filter to apply to list models. This can be
            either an ``Operator`` type filter (e.g. ``And("task == ic", "framework == pytorch")``),
            or simply a string filter which will get serialized into an Identity filter.
            (e.g. ``"task == ic"``). If this argument is not supplied, all models will be listed.
            (Default: Constant(BooleanValues.TRUE)).
        region (str): Optional. The AWS region from which to retrieve JumpStart metadata regarding
            models. (Default: JUMPSTART_DEFAULT_REGION_NAME).
        list_incomplete_models (bool): Optional. If a model does not contain metadata fields
            requested by the filter, and the filter cannot be resolved to a include/not include,
            whether the model should be included. By default, these models are omitted from results.
            (Default: False).
        list_old_models (bool): Optional. If there are older versions of a model, whether the older
            versions should be included in the returned result. (Default: False).
        list_versions (bool): Optional. True if versions for models should be returned in addition
            to the id of the model. (Default: False).
    """

    model_id_version_dict: Dict[str, List[str]] = dict()
    for model_id, version in _generate_jumpstart_model_versions(
        filter=filter, region=region, list_incomplete_models=list_incomplete_models
    ):
        if model_id not in model_id_version_dict:
            model_id_version_dict[model_id] = list()
        model_id_version_dict[model_id].append(Version(version))

    if not list_versions:
        return sorted(list(model_id_version_dict.keys()))

    if not list_old_models:
        model_id_version_dict = {
            model_id: set([max(versions)]) for model_id, versions in model_id_version_dict.items()
        }

    model_id_version_set: Set[Tuple[str, str]] = set()
    for model_id in model_id_version_dict:
        for version in model_id_version_dict[model_id]:
            model_id_version_set.add((model_id, str(version)))

    return sorted(list(model_id_version_set), key=cmp_to_key(_compare_model_version_tuples))


def _generate_jumpstart_model_versions(  # pylint: disable=redefined-builtin
    filter: Union[Operator, str] = Constant(BooleanValues.TRUE),
    region: str = JUMPSTART_DEFAULT_REGION_NAME,
    list_incomplete_models: bool = False,
) -> Generator:
    """Generate models for JumpStart, and optionally apply filters to result.

    Args:
        filter (Union[Operator, str]): Optional. The filter to apply to generate models. This can be
            either an ``Operator`` type filter (e.g. ``And("task == ic", "framework == pytorch")``),
            or simply a string filter which will get serialized into an Identity filter.
            (e.g. ``"task == ic"``). If this argument is not supplied, all models will be generated.
            (Default: Constant(BooleanValues.TRUE)).
        region (str): Optional. The AWS region from which to retrieve JumpStart metadata regarding
            models. (Default: JUMPSTART_DEFAULT_REGION_NAME).
        list_incomplete_models (bool): Optional. If a model does not contain metadata fields
            requested by the filter, and the filter cannot be resolved to a include/not include,
            whether the model should be included. By default, these models are omitted from
            results. (Default: False).
    """

    if isinstance(filter, str):
        filter = Identity(filter)

    models_manifest_list = accessors.JumpStartModelsAccessor._get_manifest(region=region)
    manifest_keys = set(models_manifest_list[0].__slots__)

    all_keys: Set[str] = set()

    model_filters: Set[ModelFilter] = set()

    for operator in _model_filter_in_operator_generator(filter):
        model_filter = operator.unresolved_value
        key = model_filter.key
        all_keys.add(key)
        model_filters.add(model_filter)

    for key in all_keys:
        if "." in key:
            raise NotImplementedError(f"No support for multiple level metadata indexing ('{key}').")

    metadata_filter_keys = all_keys - SPECIAL_SUPPORTED_FILTER_KEYS

    required_manifest_keys = manifest_keys.intersection(metadata_filter_keys)
    possible_spec_keys = metadata_filter_keys - manifest_keys

    unrecognized_keys: Set[str] = set()

    is_task_filter = SpecialSupportedFilterKeys.TASK in all_keys
    is_framework_filter = SpecialSupportedFilterKeys.FRAMEWORK in all_keys
    is_supported_model_filter = SpecialSupportedFilterKeys.SUPPORTED_MODEL in all_keys

    for model_manifest in models_manifest_list:

        copied_filter = copy.deepcopy(filter)

        manifest_specs_cached_values: Dict[str, Union[bool, int, float, str, dict, list]] = {}

        model_filters_to_resolved_values: Dict[ModelFilter, BooleanValues] = {}

        for val in required_manifest_keys:
            manifest_specs_cached_values[val] = getattr(model_manifest, val)

        if is_task_filter:
            manifest_specs_cached_values[
                SpecialSupportedFilterKeys.TASK
            ] = extract_framework_task_model(model_manifest.model_id)[1]

        if is_framework_filter:
            manifest_specs_cached_values[
                SpecialSupportedFilterKeys.FRAMEWORK
            ] = extract_framework_task_model(model_manifest.model_id)[0]

        if is_supported_model_filter:
            manifest_specs_cached_values[SpecialSupportedFilterKeys.SUPPORTED_MODEL] = Version(
                model_manifest.min_version
            ) <= Version(get_sagemaker_version())

        _populate_model_filters_to_resolved_values(
            manifest_specs_cached_values,
            model_filters_to_resolved_values,
            model_filters,
        )

        _put_resolved_booleans_into_filter(copied_filter, model_filters_to_resolved_values)

        copied_filter.eval()

        if copied_filter.resolved_value in [BooleanValues.TRUE, BooleanValues.FALSE]:
            if copied_filter.resolved_value == BooleanValues.TRUE:
                yield (model_manifest.model_id, model_manifest.version)
            continue

        if copied_filter.resolved_value == BooleanValues.UNEVALUATED:
            raise RuntimeError(
                "Filter expression in unevaluated state after using values from model manifest. "
                "Model ID and version that is failing: "
                f"{(model_manifest.model_id, model_manifest.version)}."
            )
        copied_filter_2 = copy.deepcopy(filter)

        model_specs = accessors.JumpStartModelsAccessor.get_model_specs(
            region=region,
            model_id=model_manifest.model_id,
            version=model_manifest.version,
        )

        model_specs_keys = set(model_specs.__slots__)

        unrecognized_keys -= model_specs_keys
        unrecognized_keys_for_single_spec = possible_spec_keys - model_specs_keys
        unrecognized_keys.update(unrecognized_keys_for_single_spec)

        for val in possible_spec_keys:
            if hasattr(model_specs, val):
                manifest_specs_cached_values[val] = getattr(model_specs, val)

        _populate_model_filters_to_resolved_values(
            manifest_specs_cached_values,
            model_filters_to_resolved_values,
            model_filters,
        )
        _put_resolved_booleans_into_filter(copied_filter_2, model_filters_to_resolved_values)

        copied_filter_2.eval()

        if copied_filter_2.resolved_value != BooleanValues.UNEVALUATED:
            if copied_filter_2.resolved_value == BooleanValues.TRUE or (
                BooleanValues.UNKNOWN and list_incomplete_models
            ):
                yield (model_manifest.model_id, model_manifest.version)
            continue

        raise RuntimeError(
            "Filter expression in unevaluated state after using values from model specs. "
            "Model ID and version that is failing: "
            f"{(model_manifest.model_id, model_manifest.version)}."
        )

    if len(unrecognized_keys) > 0:
        raise RuntimeError(f"Unrecognized keys: {str(unrecognized_keys)}")


def get_model_url(
    model_id: str, model_version: str, region: str = JUMPSTART_DEFAULT_REGION_NAME
) -> str:
    """Retrieve web url describing pretrained model.

    Args:
        model_id (str): The model ID for which to retrieve the url.
        model_version (str): The model version for which to retrieve the url.
        region (str): Optional. The region from which to retrieve metadata.
            (Default: JUMPSTART_DEFAULT_REGION_NAME)
    """

    model_specs = accessors.JumpStartModelsAccessor.get_model_specs(
        region=region, model_id=model_id, version=model_version
    )
    return model_specs.url
