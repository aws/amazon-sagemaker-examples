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


def get_model():

    tensor_parallel = int(os.getenv("TENSOR_PARALLEL_DEGREE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    model_dir = "/tmp/model"
    bucket = os.environ.get("MODEL_S3_BUCKET")
    key_prefix = os.environ.get("MODEL_S3_PREFIX")

    print(f"rank: {local_rank}")

    if local_rank == 0:

        if f"{model_dir}/DONE" not in glob(f"{model_dir}/*"):
            print("Starting Model downloading files")
            # download_files(s3_paths, model_dir)
            subprocess.run(
                ["aws", "s3", "cp", "--recursive", f"s3://{bucket}/{key_prefix}", model_dir]
            )
            print("Model downloading finished")

            # write file when download complete. Could use dist.barrier() but this makes it easier to check if model is downloaded in case of retry
            with open(f"{model_dir}/DONE", "w") as f:
                f.write("download_complete")

    else:
        while f"{model_dir}/DONE" not in glob(f"{model_dir}/*"):
            time.sleep(60)
            if local_rank == 1:
                print("Model download in progress")

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
