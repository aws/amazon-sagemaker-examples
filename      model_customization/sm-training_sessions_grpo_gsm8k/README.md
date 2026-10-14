# GRPO on GSM8K with Amazon SageMaker Training Sessions

A minimal example of writing your own training loop against SageMaker Training Sessions. Two
notebooks: train a LoRA adapter on GSM8K math problems with a GRPO-style reinforcement learning
loop, then deploy the result to a SageMaker real-time endpoint.

The loop is deliberately small so it can be read top to bottom. For longer-running SFT and RL
setups, see the [SageMaker Training Sessions Cookbook](https://github.com/aws/sagemaker-training-sessions-cookbook).

## Contents

```
.
├── infrastructure/
│   ├── setup_infra.py       # create the IAM execution role
│   └── cleanup_infra.py     # tear down what the example created
├── data/                    # GSM8K train/eval splits, written by sm-training_sessions_grpo_gsm8k.ipynb
├── sm-training_sessions_grpo_gsm8k.ipynb            # GRPO training on GSM8K
├── sm-training_sessions_deploy_grpo_checkpoint.ipynb           # deploy the trained checkpoint
└── requirements.txt
```

## Prerequisites

- Linux or macOS, Python 3.10, 3.11, or 3.12
- SageMaker Python SDK v3.25.0 or later (`pip install -r requirements.txt` installs it)
- AWS credentials, and quota for SageMaker Training Sessions in `us-east-1` or `us-west-2`

## Setup

```bash
pip install -r requirements.txt
python infrastructure/setup_infra.py --region us-east-1
```

`setup_infra.py` creates an IAM role named `SageMakerTrainingSessionRole`, trusted by both the
Training Sessions service and SageMaker Inference. The role itself is global, but `--region`
scopes its S3 permissions to that Region's default bucket, so use the Region you intend to
train in. Re-running for a different Region replaces those permissions rather than adding
to them.

Nothing else needs configuring. The notebooks resolve region, account, execution role, and
default S3 bucket at runtime, and create the model package group on first run.

## Run

Run `sm-training_sessions_grpo_gsm8k.ipynb`, then `sm-training_sessions_deploy_grpo_checkpoint.ipynb`.

`sm-training_sessions_grpo_gsm8k.ipynb` opens a session, measures baseline accuracy on held-out problems, runs the RL
loop, measures accuracy again, and saves two checkpoints: a merged model for deployment and a
resumable training state. `sm-training_sessions_deploy_grpo_checkpoint.ipynb` finds the merged checkpoint, hosts it on a real-time
endpoint, invokes it, and deletes the endpoint.

## Clean up

Both notebooks clean up after themselves if you run them to completion: `sm-training_sessions_grpo_gsm8k.ipynb` stops
the session, and `sm-training_sessions_deploy_grpo_checkpoint.ipynb` deletes the endpoint and its resources. What remains are the
S3 outputs, the registered checkpoints, and the IAM role.

```bash
python infrastructure/cleanup_infra.py                    # list only, deletes nothing
python infrastructure/cleanup_infra.py --stop-sessions
python infrastructure/cleanup_infra.py --delete-endpoints <endpoint-name>
python infrastructure/cleanup_infra.py --all              # sessions, S3, model package group, role
```

A training session holds GPU resources while it is running, so stop it when you are done.
