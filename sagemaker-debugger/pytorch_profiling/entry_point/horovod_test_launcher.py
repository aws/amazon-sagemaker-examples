# Standard Library
import argparse
import os
import subprocess
import sys
from distutils.util import strtobool

# Third Party
from torch.cuda import device_count

HOROVOD_PYTORCH_TEST_MNIST_SCRIPT = "./horovod_mnist.py"


HOROVOD_MNIST_SCRIPT_NAME = "horovod_mnist.py"


def launch_horovod_job(script_file_path, script_args, num_workers, smprofile_path, mode):
    command = ["mpirun", "-np", str(num_workers)] + [sys.executable, script_file_path] + script_args
    env_dict = os.environ.copy()
    env_dict["HOROVOD_TIMELINE"] = f"{smprofile_path}"
    if mode == "cpu":
        env_dict["CUDA_VISIBLE_DEVICES"] = "-1"
    subprocess.check_call(command, env=env_dict)


def main():
    parser = argparse.ArgumentParser(description="Launch horovod test")
    parser.add_argument("--script", type=str, default=HOROVOD_PYTORCH_TEST_MNIST_SCRIPT)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--gpu", type=strtobool, default=1)
    parser.add_argument("--profile_path", type=str, default="./hvd_timeline.json")
    parser.add_argument("--model", type=str, default="resnet50")
    opt = parser.parse_args()

    if opt.gpu == 1:
        mode = "gpu"
    else:
        mode = "cpu"
    num_workers = 1 if bool(device_count()) is False else device_count()
    print(f"Number of workers = {num_workers}")
    mode_args = []
    if mode == "cpu":
        mode_args += ["--use_only_cpu", "true"]
    mode_args += [
        "--epochs",
        str(opt.epoch),
        "--batch_size",
        str(opt.batch_size),
        "--model",
        str(opt.model),
    ]
    launch_horovod_job(
        script_file_path=opt.script,
        script_args=mode_args,
        num_workers=num_workers,
        smprofile_path=opt.profile_path,
        mode=mode,
    )


if __name__ == "__main__":
    main()
