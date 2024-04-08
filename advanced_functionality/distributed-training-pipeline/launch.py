import json
import os
import signal
import subprocess
import sys
import stat
from pathlib import Path

def main():


    exitcode = 0
    try:
        runtime_env = json.loads(os.environ["SM_TRAINING_ENV"])
        hyperparameters = runtime_env.get("hyperparameters")

        script_file = hyperparameters.get("script_file", "script.sh")
        torch_distributed = hyperparameters.get("torch_distributed", "false")
        
        if torch_distributed.lower() == "true":
            print("set torch distributed environment")

            hosts = runtime_env["hosts"]
            current_host = runtime_env["current_host"]

            nnodes = len(hosts)
            master_addr = runtime_env["master_hostname"]
            master_port = hyperparameters.get('master_port', 44000) 
            node_rank = hosts.index(current_host)
            num_gpus = int(runtime_env["num_gpus"])
            num_neurons = int(runtime_env["num_neurons"])
            nproc_per_node = num_neurons if num_neurons > 0 else num_gpus if num_gpus > 0 else 1

            os.environ["PET_NNODES"]=str(nnodes)
            os.environ["PET_NPROC_PER_NODE"]=str(nproc_per_node)
            os.environ["PET_NODE_RANK"]=str(node_rank)
            os.environ["PET_MASTER_ADDR"]=str(master_addr)
            os.environ["PET_MASTER_PORT"]=str(master_port)

        file = Path(__file__).resolve()
        base_dir = file.parent
    
        script_path = os.path.join(base_dir, script_file)

        print(f"chmod {script_path} to executable")
        st = os.stat(script_path)
        os.chmod(script_path, st.st_mode | stat.S_IEXEC)

        print(f"invoking script: {script_path}")
        
        process = subprocess.Popen(
            script_path,
            encoding="utf-8",
            shell=True,
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        while True:
            if process.poll() != None:
                break

            output = process.stdout.readline()
            if output:
                print(output.strip())

        exitcode = process.poll()
        print(f"Script exit code:{exitcode}")
        exitcode = 0
    except Exception as e:
        print("Script exception:", file=sys.stderr)
        exitcode = 1
        print(str(e), file=sys.stderr)

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exitcode)


if __name__ == "__main__":
    main()
