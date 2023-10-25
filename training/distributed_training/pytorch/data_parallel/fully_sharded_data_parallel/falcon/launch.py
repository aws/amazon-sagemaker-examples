import sagemaker,boto3

from sagemaker.pytorch import PyTorch
sagemaker_session = sagemaker.Session()

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']


sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sagemaker_session is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sagemaker_session.default_bucket()

sagemaker_session = sagemaker.Session()
print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sagemaker_session.default_bucket()}")
print(f"sagemaker session region: {sagemaker_session.boto_region_name}")

model_id = "tiiuae/falcon-7b"
dataset_name = "tatsu-lab/alpaca"

from datasets import load_dataset
from transformers import AutoTokenizer 

# Load Tokenizer 

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load dataset from huggingface.co
dataset = load_dataset(dataset_name)

# downsample dataset to 10k
dataset = dataset.shuffle(42)


if "validation" not in dataset.keys():
    dataset["validation"] = load_dataset(
        dataset_name,
        split="train[:5%]"
    )

    dataset["train"] = load_dataset(
        dataset_name,
        split="train[5%:]"
    )

from itertools import chain
from functools import partial

def group_texts(examples,block_size = 2048):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

column_names = dataset["train"].column_names

lm_dataset = dataset.map(
    lambda sample: tokenizer(sample["text"],return_token_type_ids=False), batched=True, remove_columns=list(column_names)
).map(
    partial(group_texts, block_size=2048),
    batched=True,
)

training_input_path = f'processed/data/'
lm_dataset.save_to_disk(training_input_path)

print(f"Saved data to: {training_input_path}")

training_input_path = f's3://{sagemaker_session.default_bucket()}/processed/data/'
print(f"training dataset to: {training_input_path}")# save train_dataset to s3
lm_dataset.save_to_disk(training_input_path)

print(f"uploaded data to: {training_input_path}")
import time


ecr_image="570106654206.dkr.ecr.us-west-2.amazonaws.com/unified-herring:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker"
job_name = f'huggingface-fsdp-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
subnet_config = ["subnet-02bb299d2e6af1d47"]
security_group_config = ["sg-051fa281c203e03a9"]

# hyperparameters, which are passed into the training job
hyperparameters={
    'model_id': model_id, # model id from huggingface.co/models
    'dataset_path': '/opt/ml/input/data/train', # path where sagemaker will save training dataset
    'valid_path':"/opt/ml/input/data/valid",
    'gradient_checkpointing': True, # enable gradient checkpointing
    'bf16': True, # enable mixed precision training
    'optimizer': "adamw_torch", # optimizer
    'per_device_train_batch_size': 1, # batch size per device during training
    'epochs': 1, # number of epochs to train
    'fsdp': '"full_shard auto_wrap"', # fully sharded data parallelism
    'cache_dir': "/opt/ml/sagemaker/warmpoolcache", #change this to /tmp if not using warmpools
    'max_steps':30
}

estimator = PyTorch(
  entry_point="train.py",
  max_run=1800,
  job_name=job_name,
  role=role,
  framework_version="2.0.1",
  py_version="py310",
  image_uri=ecr_image,
  source_dir="./scripts",
  instance_count=2,
  instance_type="ml.p4d.24xlarge",
  sagemaker_session=sagemaker_session,
  subnets=subnet_config,
  security_group_ids=security_group_config,
  disable_output_compression=True,
  distribution={"torch_distributed": {"enabled": True}},
  keep_alive_period_in_seconds=1800,
  hyperparameters=hyperparameters,
)
data = {'train': training_input_path}

estimator.fit(data, wait=True)
