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
"""Configuration for the SageMaker Training Compiler."""
from __future__ import absolute_import
import logging

from sagemaker.workflow import is_pipeline_variable

logger = logging.getLogger(__name__)


class TrainingCompilerConfig(object):
    """The SageMaker Training Compiler configuration class."""

    DEBUG_PATH = "/opt/ml/output/data/compiler/"
    SUPPORTED_INSTANCE_CLASS_PREFIXES = ["p3", "p3dn", "g4dn", "p4d", "g5"]

    HP_ENABLE_COMPILER = "sagemaker_training_compiler_enabled"
    HP_ENABLE_DEBUG = "sagemaker_training_compiler_debug_mode"

    def __init__(self, enabled=True, debug=False):
        """This class initializes a ``TrainingCompilerConfig`` instance.

        `Amazon SageMaker Training Compiler
        <https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler.html>`_
        is a feature of SageMaker Training
        and speeds up training jobs by optimizing model execution graphs.

        You can compile Hugging Face models
        by passing the object of this configuration class to the ``compiler_config``
        parameter of the :class:`~sagemaker.huggingface.HuggingFace`
        estimator.

        Args:
            enabled (bool): Optional. Switch to enable SageMaker Training Compiler.
                The default is ``True``.
            debug (bool): Optional. Whether to dump detailed logs for debugging.
                This comes with a potential performance slowdown.
                The default is ``False``.

        **Example**: The following code shows the basic usage of the
        :class:`sagemaker.huggingface.TrainingCompilerConfig()` class
        to run a HuggingFace training job with the compiler.

        .. code-block:: python

            from sagemaker.huggingface import HuggingFace, TrainingCompilerConfig

            huggingface_estimator=HuggingFace(
                ...
                compiler_config=TrainingCompilerConfig()
            )

        .. seealso::

            For more information about how to enable SageMaker Training Compiler
            for various training settings such as using TensorFlow-based models,
            PyTorch-based models, and distributed training,
            see `Enable SageMaker Training Compiler
            <https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler-enable.html>`_
            in the `Amazon SageMaker Training Compiler developer guide
            <https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler.html>`_.

        """

        self.enabled = enabled
        self.debug = debug

        self.disclaimers_and_warnings()

    def __nonzero__(self):
        """Evaluates to 0 if SM Training Compiler is disabled."""
        return self.enabled

    def disclaimers_and_warnings(self):
        """Disclaimers and warnings.

        Logs disclaimers and warnings about the
        requested configuration of SageMaker Training Compiler.

        """

        if self.enabled and self.debug:
            logger.warning(
                "Debugging is enabled."
                "This will dump detailed logs from compilation to %s"
                "This might impair training performance.",
                self.DEBUG_PATH,
            )

    def _to_hyperparameter_dict(self):
        """Converts configuration object into hyperparameters.

        Returns:
            dict: A portion of the hyperparameters passed to the training job as a dictionary.

        """

        compiler_config_hyperparameters = {
            self.HP_ENABLE_COMPILER: self.enabled,
            self.HP_ENABLE_DEBUG: self.debug,
        }

        return compiler_config_hyperparameters

    @classmethod
    def validate(cls, estimator):
        """Checks if SageMaker Training Compiler is configured correctly.

        Args:
            estimator (:class:`sagemaker.estimator.Estimator`): An estimator object.
                When SageMaker Training Compiler is enabled, it validates if
                the estimator is configured to be compatible with Training Compiler.


        Raises:
            ValueError: Raised if the requested configuration is not compatible
                        with SageMaker Training Compiler.
        """
        if is_pipeline_variable(estimator.instance_type):
            warn_msg = (
                "Estimator instance_type is a PipelineVariable (%s), "
                "which has to be interpreted as one of the "
                "%s classes in execution time."
            )
            logger.warning(
                warn_msg,
                type(estimator.instance_type),
                str(cls.SUPPORTED_INSTANCE_CLASS_PREFIXES).replace(",", ""),
            )
        elif estimator.instance_type:
            if "local" not in estimator.instance_type:
                requested_instance_class = estimator.instance_type.split(".")[
                    1
                ]  # Expecting ml.class.size
                if not any(
                    [requested_instance_class == i for i in cls.SUPPORTED_INSTANCE_CLASS_PREFIXES]
                ):
                    error_helper_string = (
                        "Unsupported Instance class {}."
                        "SageMaker Training Compiler only supports {}"
                    )
                    error_helper_string = error_helper_string.format(
                        requested_instance_class, cls.SUPPORTED_INSTANCE_CLASS_PREFIXES
                    )
                    raise ValueError(error_helper_string)
            elif estimator.instance_type == "local":
                error_helper_string = (
                    "SageMaker Training Compiler doesn't support local mode."
                    "It only supports the following GPU instances: {}"
                )
                error_helper_string = error_helper_string.format(
                    cls.SUPPORTED_INSTANCE_CLASS_PREFIXES
                )
                raise ValueError(error_helper_string)

        if estimator.distribution and "smdistributed" in estimator.distribution:
            raise ValueError(
                "SageMaker distributed training configuration is currently not compatible with "
                "SageMaker Training Compiler."
            )

        if estimator.debugger_hook_config or (not estimator.disable_profiler):
            helper_string = (
                "Using Debugger and/or Profiler with SageMaker Training Compiler "
                "might add recompilation overhead and degrade"
                "performance. Found debugger_hook_config={} "
                "disable_profiler={}. Please set "
                "debugger_hook_config=None and disable_profiler=True for optimal "
                "performance. For more information, see Training Compiler "
                "Performance Considerations "
                "(https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler-tips-pitfalls.html"
                "#training-compiler-tips-pitfalls-considerations)."
            )
            helper_string = helper_string.format(
                estimator.debugger_hook_config, estimator.disable_profiler
            )
            logger.warning(helper_string)

        if estimator.instance_groups:
            raise ValueError(
                "SageMaker Training Compiler currently only supports homogeneous clusters of "
                "the following GPU instance families: {}. Please use the 'instance_type' "
                "and 'instance_count' parameters instead of 'instance_groups'".format(
                    cls.SUPPORTED_INSTANCE_CLASS_PREFIXES
                )
            )
