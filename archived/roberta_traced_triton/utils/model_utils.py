"""Utility functions for saving an compiling models"""

import boto3
import torch
from pathlib import Path
from typing import Union, List
from jinja2 import Environment, FileSystemLoader
import sagemaker
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
import shutil
import subprocess
import tarfile
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers.onnx import FeaturesManager
import time
import timm


def get_model_from_torch_hub(repo: str, model_name: str, pretrained:bool=True) -> torch.nn.Module:
    model = torch.hub.load(repo, model_name, pretrained=pretrained)
    model.eval()
    return model

def get_model_from_hf_hub(model_name:str):
    """returns a model and tokenizer from HF Hub"ArithmeticError"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torchscript=True)
    model = model.eval()

    return tokenizer, model

def get_model_from_timm(model_name: str, pretrained:bool=True) -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval()
    return model

def count_parameters(model):
    """Provides the number of parameters for the given model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_pt_jit(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    save_path: Union[str, Path] = ".",
) -> Path:
    jit_model = torch.jit.trace(model, sample_input)
    save_path = Path(save_path) / "model.pt"
    jit_model.save(save_path)
    return save_path


def export_onnx(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    save_path: Union[str, Path] = ".",
) -> Path:
    
    """Export PyTorch Model to ONNX"""

    save_path = Path(save_path) / "model.onnx"

    torch.onnx.export(
        model,
        sample_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["INPUT__0"],
        output_names=["OUTPUT__0"],
        dynamic_axes={"INPUT__0": {0: "batch_size"}, "OUTPUT__0": {0: "batch_size"}},
    )

    return save_path

def export_onnx_nlp(model, tokenizer, save_path: Union[str, Path] = ".",):
    
    """Exports and onnx using the Hugging Face Transformers onnx utility

    Returns:
        Tuple[Path, Dict]: save path of the onnx artifact and its configuration
    """
    
    save_path = Path(save_path) / "model.onnx"
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model)
    onnx_config = model_onnx_config(model.config)

    onnx_inputs, onnx_outputs = transformers.onnx.export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=12,
        output=save_path
    )
    
    return save_path, onnx_config


def compile_trt_sm_proc(
    onnx_model_path: Union[str, Path],
    role: str,
    image_uri: str,
    instance_type: str,
    trt_compilation_args: List[str],
    save_path: Union[str, Path] = ".",
    wait=True,
    verbose=False,
):
    """Compile an ONNX Model to TensorRT" using a SageMaker Processing Job"""

    onnx_model_path = Path(onnx_model_path)

    code_path = Path.cwd() / "trt_compile.sh"
    code_path.open("w").write("/usr/src/tensorrt/bin/trtexec $@")

    if (not wait) & verbose:
        raise Exception("vebose=True option can only be used with wait=True")

    compilation_job = ScriptProcessor(
        image_uri=image_uri,
        command=["/bin/bash"],
        instance_type=instance_type,
        volume_size_in_gb=100,
        instance_count=1,
        base_job_name="trt-compilation",
        role=role,
    )

    arguments = [
        f"--onnx=/opt/ml/processing/input_data/{onnx_model_path.name}",
        "--saveEngine=/opt/ml/processing/output/model.plan",
    ]

    arguments += trt_compilation_args

    print("Running sagemaker processing job to compile model to TensorRT")
    compilation_job.run(
        code="trt_compile.sh",
        inputs=[
            ProcessingInput(
                source=str(onnx_model_path),
                destination="/opt/ml/processing/input_data/",
            ),
        ],
        outputs=[ProcessingOutput(source="/opt/ml/processing/output/")],
        arguments=arguments,
        wait=(wait & verbose),
    )

    if wait:
        job_waiter = boto3.client("sagemaker").get_waiter(
            "processing_job_completed_or_stopped"
        )
        print("Waiting for job to complete")
        job_waiter.wait(ProcessingJobName=compilation_job.latest_job.job_name)
        print("Job completed")

        s3_output = compilation_job.latest_job.outputs[0].destination
        save_path = Path(save_path)
        cmd = f"aws s3 cp --recursive {s3_output} {str(save_path)}"
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        out, _ = p.communicate()
        print(out.decode("utf8"))
        return save_path / "model.plan"

    else:
        return compilation_job
    
    
def generate_triton_config(
    platform: str,
    triton_inputs=[],
    triton_outputs=[],
    save_path: Union[str, Path] = ".",
    template_path='config_templates/'
):
    """Generates a Triton configuration config.pbtxt file 

    Args:
        platform (str): pt or trt
        triton_inputs (list, optional): model input configuration. Defaults to [].
        triton_outputs (list, optional): model output configuration. Defaults to [].
        save_path (Union[str, Path], optional): Path to where the generated file will be saved. Defaults to ".".

    Returns:
        Path: Path to the config.pbtxt file
    """

    environment = Environment(loader=FileSystemLoader(template_path))
    template = environment.get_template(f"{platform}_nlp_config.pbtxt")
    config = template.render(inputs=triton_inputs, outputs=triton_outputs)
    config_path =(Path(save_path) / "config.pbtxt")
    config_path.open("w").write(config)

    return config_path

def gen_trt_inp_compilation_config(onnx_config, max_seq_len, batch_sizes=[1,16,32]):
    
    """Generates the compilation arguments for TensorRT conversion.
       batch_sizes represent the minShapes, optShapes, and maxShapes for the compilation 

    Returns:
        _type_: _description_
    """
    
    trt_input_shape_args = []
    for shape_config, batch_size in zip(
        ["minShapes", "optShapes", "maxShapes"], batch_sizes
    ):
        trt_input_shape_args.append(
            f"--{shape_config}="
            + ",".join(
                [
                    f"{input_name}:{batch_size}x{max_seq_len}"
                    for input_name in onnx_config.inputs
                ]
            )
        )
    return trt_input_shape_args

def compile_trt(
    onnx_model_path: Union[str, Path],
    sagemaker_session: sagemaker.Session(),
    bucket:str,
    prefix:str,
    role: str,
    image_uri: str,
    instance_type: str,
    trt_compilation_args: List[str],
    save_path: Union[str, Path] = ".",
    logs:bool=False,
    wait:bool=True,
):
    """Compile an ONNX Model to TensorRT" using a SageMaker Training Job"""
    
    sm_client = boto3.client("sagemaker")
    compilation_job_name =  f"trt-compilation" + time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.gmtime()
    )
    
    onnx_model_s3_path = sagemaker_session.upload_data(onnx_model_path.as_posix(), bucket=bucket, key_prefix=f"{prefix}/{compilation_job_name}/input")
    s3_output_path = f"s3://{bucket}/{prefix}"
    
    response = sm_client.create_training_job(
    TrainingJobName=compilation_job_name,

    AlgorithmSpecification={
        'TrainingImage': image_uri,
        'TrainingInputMode': 'File',
        'ContainerArguments': [
            '/usr/src/tensorrt/bin/trtexec', f'--onnx=/opt/ml/input/data/model/{onnx_model_path.name}', "--saveEngine=/opt/ml/model/model.plan"
        ] +  trt_compilation_args
    },
    RoleArn=role,
    InputDataConfig=[
                {
                    'ChannelName': 'model',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': onnx_model_s3_path,
                            'S3DataDistributionType': 'FullyReplicated',

                        },

                    },

                    'InputMode': 'File',
                },
            ],
            OutputDataConfig={
                'S3OutputPath': s3_output_path
            },
            ResourceConfig={
                'InstanceType': instance_type,
                'VolumeSizeInGB': 100,
                'InstanceCount' : 1
            },

            StoppingCondition={
            'MaxRuntimeInSeconds': 7200
            }



        )
    
    if logs == True:
        print(f"Waiting for training job {compilation_job_name} to complete")
        sagemaker_session.logs_for_job(compilation_job_name, wait=True)
    elif wait == True:
        print(f"Waiting for training job {compilation_job_name} to complete")
        sagemaker_session.wait_for_job(compilation_job_name)
    else:
        print(f"Not waiting for job. Compiled trt artifact can be downloaded from s3://{bucket}/{prefix}/{compilation_job_name}/output/model.tar.gz once the job is complete")
        return compilation_job_name
        
    sagemaker_session.download_data("tmp_model", bucket = bucket, key_prefix=f"{prefix}/{compilation_job_name}/output/model.tar.gz")
    save_path = Path(save_path)
    with tarfile.open("tmp_model/model.tar.gz", "r:gz") as tar:
        tar.extractall(save_path)

    shutil.rmtree("tmp_model/")

    return save_path / "model.plan"


def package_triton_model(
    model_name: str, model_file_path: Union[str, Path], config_path: Union[str, Path]
):
    """Generates the model.tar.gz model package

    Args:
        model_name (str): name of the model
        model_file_path (Union[str, Path]): Location of the serialized model file
        config_path (Union[str, Path]): Location of the config.pbtxt file

    Returns:
        Path: The local path of the model.tar.gz file
    """

    model_file_path = Path(model_file_path)
    config_path = Path(config_path)
    model_tar_path = Path(f"{model_name}.tar.gz")

    tar = tarfile.open(model_tar_path, "w:gz")
    tar.add(model_file_path, f"1/{model_file_path.name}")
    tar.add(config_path, "config.pbtxt")
    tar.close()

    return model_tar_path