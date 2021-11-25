import subprocess
import sys

if __name__ == "__main__":
    arguments_command = " ".join([arg for arg in sys.argv[1:]])
    """
    The following line will take care of setting up inter node communication as well as managing intra node workers for each GPU.
    """
    subprocess.check_call("python -m torch_xla.distributed.sm_dist " + arguments_command, shell=True)
