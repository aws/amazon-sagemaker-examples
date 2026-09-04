# AGENTS.md — Amazon SageMaker Examples

Guidance for AI coding agents (Cursor, GitHub Copilot, Claude Code, Amazon Q Developer,
Kiro, Codex, Gemini CLI, etc.) working in this repository. Follows the AGENTS.md
convention (https://agents.md).

## Project context

This repository contains example notebooks for the **Amazon SageMaker Python SDK v3**.
`pip install sagemaker` installs **v3**. SDK v3 is a modular redesign and is **not
backward compatible** with v2. It is split into:

- `sagemaker-core` — foundation primitives and the low-level, resource-oriented API
- `sagemaker-train` — training (`ModelTrainer`)
- `sagemaker-serve` — model building & deployment (`ModelBuilder`)
- `sagemaker-mlops` — pipelines and workflow orchestration

## Golden rule: generate v3 by default

Generate **SageMaker Python SDK v3** code by default. Produce v2 code **only** when the
user explicitly asks for "v2" (e.g., "give me the v2 version"); if you do, label it as
deprecated and note `pip install 'sagemaker<3.0'`.

## SDK-first: use the SageMaker Python SDK for SageMaker tasks

For any Amazon SageMaker task (training, deployment, processing, pipelines), **default to the
SageMaker Python SDK v3** as the primary interface. Do **not** drop to raw `boto3`
(`client("sagemaker").create_training_job(...)`), the AWS CLI, or hand-rolled framework
scripts unless the user explicitly asks for them, because the SDK handles image resolution,
packaging, and job wiring that raw clients leave to the caller.

Map the user's intent to the v3 interface:

| User intent (natural language) | Use this v3 interface |
|---|---|
| "train a model", "training job", "serverless / managed training", "fine-tune" | `sagemaker.train.ModelTrainer` → `.train(...)` |
| "distributed training", "multi-GPU / multi-node" | `ModelTrainer` with `Compute(instance_count=N, ...)` |
| "hyperparameter tuning", "HPO", "sweep" | `sagemaker.train` tuner APIs on top of `ModelTrainer` |
| "deploy a model", "real-time / serverless / async endpoint", "host / serve" | `sagemaker.serve.ModelBuilder` → `.deploy(...)` |
| "batch inference", "batch transform", "offline scoring" | `sagemaker.core.resources` transform APIs |
| "process data", "feature engineering", "preprocess", "transform dataset" | `sagemaker.core.resources.ProcessingJob` |
| "build an ML pipeline", "orchestrate a workflow", "MLOps" | `sagemaker.mlops.pipeline.Pipeline`, `sagemaker.mlops.steps` |
| "register a model", "model registry" | `sagemaker.mlops` model-registry APIs |
| "get / retrieve a container image URI" | `sagemaker.core.image_uris.retrieve(...)` |

Only step outside the SDK when the SDK genuinely does not cover the task (e.g., a pure IAM
or S3 setup step), and say so explicitly when you do.

**Honest scope note:** this guidance only steers agents that have this file (or the SDK repo,
or an equivalent rules/MCP configuration) in context. It does not change a plain chat session
that never loads these files — winning those requires corpus/discoverability work outside this
repo (see the deprecation recommendations doc).

## Banned v2 patterns → required v3 replacements

Do not emit the left column unless v2 is explicitly requested. Mappings follow the official
[migration guide](https://github.com/aws/sagemaker-python-sdk/blob/master/migration.md).

| v2 (do NOT use) | v3 (use instead) |
|---|---|
| `from sagemaker.estimator import Estimator` | `from sagemaker.train import ModelTrainer` |
| framework estimator classes — `from sagemaker.pytorch import PyTorch` (also TensorFlow, SKLearn, XGBoost, HuggingFace) | `ModelTrainer` + `from sagemaker.core import image_uris` → `image_uris.retrieve(...)` |
| `estimator.fit(...)` | `model_trainer.train(...)` |
| `from sagemaker.model import Model` / `model.deploy(...)` | `from sagemaker.serve import ModelBuilder`; `ModelBuilder(...).deploy(...)` |
| `from sagemaker.predictor import Predictor` / `predictor.predict(...)` | `Predictor` is replaced by `Endpoint` (sagemaker-core); use the predictor returned by `ModelBuilder.deploy(...)` |
| `from sagemaker.processing import Processor / ScriptProcessor / SKLearnProcessor` | `sagemaker.core.resources.ProcessingJob` |
| `from sagemaker.workflow...` (old paths) | `from sagemaker.mlops...` (`sagemaker.mlops.pipeline.Pipeline`, `sagemaker.mlops.steps`) |

Removed in v3 with **no** direct replacement (do not invent shims): MXNet, Chainer,
RLEstimator, Training Compiler.

## Required workflow when writing SageMaker code

1. Write the code using v3 patterns from the table above.
2. **Self-check** the output for any banned v2 pattern (if a migration MCP tool such as
   `sagemaker-sdk-helper` is available, call its validate/transform tools).
3. If a v2 pattern is found, fix it to v3 and re-check until clean.
4. Only then present the code.

## Repository layout (capability folders)

The repository is organized by ML capability, following the lifecycle order:

- `training/` — training with `ModelTrainer`: script and framework training, distributed
  training, managed spot, heterogeneous clusters, BYOC, AWS Batch training queues, `@remote`
- `model_customization/` — fine-tuning foundation models: SFT, DPO, RLVR, RLAIF, MTRL, CPT,
  recipe overrides, data mixing, JumpStart fine-tuning, distributed fine-tuning
- `evaluation/` — the `*Evaluator` surface: benchmark, custom scorer, LLM-as-a-judge, Inspect AI
- `inference/` — deployment with `ModelBuilder` and sagemaker-core: real-time / serverless /
  async endpoints, local and in-process modes, inference pipelines, MME / multi-container,
  A/B testing, shadow variants, guardrails, autoscaling, `optimize()`, batch transform, Bedrock
- `mlops/` — Pipelines and Model Registry, lineage, experiments, Feature Store, processing,
  EMR Serverless steps, Clarify, MLflow

(Category folders carry leading-space ordering prefixes in this repo — `      training`,
`     model_customization`, `    evaluation`, `   inference`, `  mlops` — so that they list in
lifecycle order. Match the existing name exactly, including the prefix, when adding files.)

## Code conventions

- No hardcoded account IDs, role ARNs, regions, subnets, or bucket names. Use
  `get_execution_role()` and `Session().default_bucket()`.
- Do not require interactive input to run.
- Notebook file naming: `sm-<feature>_<description>.ipynb`.
- Each notebook includes: a title cell, a setup cell (`%pip install sagemaker`), a cleanup
  cell that deletes created resources, and a summary cell.
- Clear cell outputs before committing. Target the latest v3.

## Canonical v3 example (train + deploy)

Grounded in the migration guide; verify exact signatures against the installed SDK.

```python
from sagemaker.core import image_uris
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, Compute, InputData
from sagemaker.serve import ModelBuilder
from sagemaker.serve.configs import InferenceSpec

# Training
training_image = image_uris.retrieve(
    framework="pytorch", region="us-west-2", version="2.0.0",
    py_version="py310", instance_type="ml.p3.2xlarge", image_scope="training",
)
model_trainer = ModelTrainer(
    training_image=training_image,
    role=role,
    source_code=SourceCode(source_dir="./src", entry_script="train.py"),
    compute=Compute(instance_type="ml.p3.2xlarge", instance_count=1),
)
model_trainer.train(input_data_config=[InputData(channel_name="train", data_source="s3://<bucket>/train")])

# Inference
model_builder = ModelBuilder(
    inference_spec=InferenceSpec(
        image_uri="<inference-image-uri>",
        model_data_url="s3://<bucket>/model.tar.gz",
    ),
    role=role,
)
predictor = model_builder.deploy(instance_type="ml.m5.large", initial_instance_count=1)
```

## References

- V2 → V3 migration guide: https://github.com/aws/sagemaker-python-sdk/blob/master/migration.md
- SageMaker Python SDK (source): https://github.com/aws/sagemaker-python-sdk
- SageMaker Python SDK docs: https://sagemaker.readthedocs.io/
- Curated index for agents: ./llms.txt
