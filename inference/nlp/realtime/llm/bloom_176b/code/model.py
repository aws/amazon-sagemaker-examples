from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from djl_python import Input, Output
import os
import deepspeed
import torch
import torch.distributed as dist
import sys
import subprocess
import time
from glob import glob


tokenizer = None
model = None


def check_config():
    local_rank = os.getenv("LOCAL_RANK")
    curr_pid = os.getpid()
    print(
        f"__Number CUDA Devices:{torch.cuda.device_count()}:::local_rank={local_rank}::curr_pid={curr_pid}::"
    )

    if not local_rank:
        return False

    return True


def get_model():

    if not check_config():
        raise Exception(
            "DJL:DeepSpeed configurations are not default. This code does not support non default configurations"
        )

    deepspeed.init_distributed("nccl")

    tensor_parallel = int(os.getenv("TENSOR_PARALLEL_DEGREE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    model_dir = "/tmp/model"
    bucket = os.environ.get("MODEL_S3_BUCKET")
    key_prefix = os.environ.get("MODEL_S3_PREFIX")
    curr_pid = os.getpid()
    print(f"tensor_parallel={tensor_parallel}::curr_pid={curr_pid}::")
    print(
        f"Current Rank: {local_rank}:: pid={curr_pid}::Going to load the model weights on rank 0: bucket={bucket}::key={key_prefix}::"
    )

    if local_rank == 0:

        if f"{model_dir}/DONE" not in glob(f"{model_dir}/*"):
            print("Starting Model downloading files pid={curr_pid}::")
            print(f"Starting Model pid={curr_pid}::")

            try:
                # --
                proc_run = subprocess.run(
                    ["aws", "s3", "cp", "--recursive", f"s3://{bucket}/{key_prefix}", model_dir],
                    capture_output=True,
                    text=True,
                )  # python 7 onwards
                print(f"Model download finished: pid={curr_pid}::")

                # write file when download complete. Could use dist.barrier() but this makes it easier to check if model is downloaded in case of retry
                with open(f"{model_dir}/DONE", "w") as f:
                    f.write("download_complete")

                print(
                    f"Model download checkmark written out pid={curr_pid}::return_code:{proc_run.returncode}:stderr:-- >:{proc_run.stderr}"
                )
                proc_run.check_returncode()  # to throw the error in case there was one

            except subprocess.CalledProcessError as e:
                print(
                    "Model download failed: Error:\nreturn code: ",
                    e.returncode,
                    "\nOutput: ",
                    e.stderr,
                )
                raise  # FAIL FAST

    dist.barrier()  # - to ensure all processes load fine

    print(f"Load the Model  pid={curr_pid}::")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # has to be FP16 as Int8 model loading not yet supported
    with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(model_dir), torch_dtype=torch.bfloat16
        )

    model = model.eval()

    model = deepspeed.init_inference(
        model,
        mp_size=tensor_parallel,
        dtype=torch.int8,
        base_dir=model_dir,
        checkpoint=os.path.join(model_dir, "ds_inference_config.json"),
        replace_method="auto",
        replace_with_kernel_inject=True,
    )

    model = model.module
    dist.barrier()
    return model, tokenizer


def handle(inputs: Input):
    print("Model In handle")
    global model, tokenizer
    if not model:
        model, tokenizer = get_model()

    if inputs.is_empty():
        print("Model warm up: inputs were empty:called by Model server to warmup")
        # Model server makes an empty call to warmup the model on startup
        return None

    inputs = inputs.get_as_json()

    print(inputs)
    data = inputs["input"]
    generate_kwargs = inputs.get("gen_kwargs", {})
    padding = inputs.get("padding", False)

    input_tokens = tokenizer(data, return_tensors="pt", padding=padding)
    print(input_tokens)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

    with torch.no_grad():
        output = model.generate(**input_tokens, **generate_kwargs)

    print(output)

    output = tokenizer.batch_decode(output, skip_special_tokens=True)
    return Output().add_as_json(output)
