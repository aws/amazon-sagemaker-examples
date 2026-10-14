# GRPO on GSM8K with Amazon SageMaker AI training session APIs

This example writes a GRPO reinforcement learning loop against SageMaker AI training session
APIs. It has two notebooks: the first trains a LoRA adapter for GPT-OSS 20B on GSM8K math
problems, and the second deploys the trained checkpoint to a SageMaker real-time endpoint.

## About training session APIs

Fine-tuning foundation models usually forces a tradeoff between control and operational
simplicity. Managed fine-tuning services handle the infrastructure but limit you to a fixed set of
techniques. Working at a lower level gives you algorithmic freedom, but you also take on GPU
clusters, distributed training frameworks, and fault recovery.

A training session is a live, GPU-backed session that you drive from your own Python process.
Your loop decides each step, and SageMaker AI handles the GPUs, base model loading, LoRA adapter
registration, distributed execution, fault recovery, and checkpoint management. You can implement
any training algorithm: supervised fine-tuning, reinforcement learning with custom reward
functions, multi-stage pipelines, or new techniques. The session is stateful, so the model stays
in memory across steps, and RL workflows can sample from the current policy without a separate
inference endpoint.

## How it works

You write the training logic in Python, in a notebook or on your local machine, using three
primitives:

- `forward_backward` computes gradients over a batch
- `optim_step` applies the optimizer update
- `sample` generates model responses from the current policy

The loss function is an argument to each training call: `cross_entropy` for supervised
fine-tuning, and `ppo`, `importance_sampling`, or `cispo` for RL. The same session and API run
either kind of training.

This example uses all three for RL. In each step, the notebook samples several responses per
question with `sample`, scores them locally against the reference answer, converts the scores
into group-relative advantages, and runs `forward_backward` with the `ppo` loss followed by
`optim_step`. It then calls `save_weights_and_get_sampling_client` so the next step samples from
the updated policy.

## Quickstart

1. Install the dependencies. `requirements.txt` installs the SageMaker Python SDK v3.25.0 or
   later, which includes the training session APIs (`sagemaker.train.training_session`):

   ```bash
   pip install -r requirements.txt
   ```

2. Create the IAM execution role:

   ```bash
   python infrastructure/setup_infra.py --region us-west-2
   ```

   `setup_infra.py` creates a role named `SageMakerTrainingSessionRole`, trusted by both the
   training sessions service and SageMaker Inference. The role is global, but `--region` scopes
   its S3 permissions to that Region's default bucket, so use the Region you train in. Running
   it again for a different Region replaces those permissions.

3. Run `sm-training_sessions_grpo_gsm8k.ipynb`, then
   `sm-training_sessions_deploy_grpo_checkpoint.ipynb`.

The notebooks resolve the Region, account, execution role, and default S3 bucket at runtime, and
create the model package group on first run.

## The core API

```python
from sagemaker.train.training_session import ServiceClient, AdamParams, SamplingParams

svc = ServiceClient(region=..., role_arn=..., model_package_group_arn=..., s3_output_path=...)
tc = svc.create_lora_training_client("openai/gpt-oss-20b", rank=32, alpha=64)
sc = tc.create_sampling_client()

# Sample from the current policy
out = sc.sample(prompt_tokens, num_samples=8, sampling_params=SamplingParams(max_tokens=512)).result()

# Train
fb = tc.forward_backward(datums, loss_fn="ppo")         # returns a future
opt = tc.optim_step(AdamParams(learning_rate=1e-5))     # pipelined behind it
fb.result(), opt.result()

# Save current weights and sample from them
sc = tc.save_weights_and_get_sampling_client()
logprobs = sc.compute_logprobs(prompt_tokens).result()

# Or sample any saved checkpoint directly
sc = svc.create_sampling_client(model_package_arn=...)

# Checkpoints
tc.save_weights_for_sampler(merge_weights=True).result()   # merged, deployable model
print(tc.save_state().result().model_package_arn)          # resumable state

tc.stop()
```

Every GPU-bound call returns a future immediately, so a step can submit all of its sampling
requests before waiting on any of them. Each operation also has an `*_async` variant, such as
`sample_async` and `forward_backward_async`.

## Contents

```
.
├── infrastructure/
│   ├── setup_infra.py                                # create the IAM execution role
│   └── cleanup_infra.py                              # remove what the example created
├── data/                                             # GSM8K splits, written by the training notebook
├── sm-training_sessions_grpo_gsm8k.ipynb             # GRPO training on GSM8K
├── sm-training_sessions_deploy_grpo_checkpoint.ipynb # deploy the trained checkpoint
└── requirements.txt
```

`sm-training_sessions_grpo_gsm8k.ipynb` opens a session, measures baseline accuracy on held-out
problems, runs the RL loop, measures accuracy again, and saves two checkpoints: a merged model
for deployment and a resumable training state.
`sm-training_sessions_deploy_grpo_checkpoint.ipynb` finds the merged checkpoint, hosts it on a
real-time endpoint, invokes it, and deletes the endpoint.

## Prerequisites

- Linux or macOS, Python 3.10, 3.11, or 3.12
- SageMaker Python SDK v3.25.0 or later
- AWS credentials, and quota for training session APIs in US East (N. Virginia), US West
  (Oregon), or Asia Pacific (Mumbai)

## Clean up

If you run both notebooks to completion, the training notebook stops the session and the deploy
notebook deletes the endpoint and its resources. The S3 outputs, the registered checkpoints, and
the IAM role remain.

```bash
python infrastructure/cleanup_infra.py                    # list only, deletes nothing
python infrastructure/cleanup_infra.py --stop-sessions
python infrastructure/cleanup_infra.py --delete-endpoints <endpoint-name>
python infrastructure/cleanup_infra.py --all              # sessions, S3, model package group, role
```

A training session holds GPU resources while it is running, so stop it when you are done.

## Further reading

- [SageMaker training sessions cookbook](https://github.com/aws/sagemaker-training-sessions-cookbook):
  complete training loops built on training session APIs, including supervised fine-tuning on chat
  datasets, RL on GSM8K, RL on competitive programming graded by sandboxed code execution, and
  KTO preference alignment, with checkpoint resume, validation, and MLflow metrics.
- [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk): source for the
  `sagemaker.train.training_session` module.
- [SageMaker AI model customization](https://aws.amazon.com/sagemaker/ai/model-customization/):
  the managed serverless customization options.
