# Standard Library
import os
import subprocess
import sys
import json


def train():
    import pprint

    pprint.pprint(dict(os.environ), width=1)
    hyperparamters = json.loads(os.environ['SM_HPS'])
    

    run_cmd = f"""python tools/train_mlperf.py --config-file 'configs/e2e_mask_rcnn_R_50_FPN_1x.yaml' \
             DTYPE 'float16' \
             PER_EPOCH_EVAL False \
             PATHS_CATALOG 'maskrcnn_benchmark/config/paths_catalog_sagemaker.py' \
             DISABLE_REDUCED_LOGGING True \
             SOLVER.BASE_LR {hyperparamters['BASE_LR']} \
             SOLVER.WEIGHT_DECAY {hyperparamters['WEIGHT_DECAY']} \
             SOLVER.MAX_ITER {hyperparamters['MAX_ITER']} \
             SOLVER.WARMUP_FACTOR {hyperparamters['WARMUP_FACTOR']} \
             SOLVER.WARMUP_ITERS {hyperparamters['WARMUP_ITERS']} \
             SOLVER.WEIGHT_DECAY_BIAS 0 \
             SOLVER.WARMUP_METHOD mlperf_linear \
             SOLVER.IMS_PER_BATCH {hyperparamters['TRAIN_IMS_PER_BATCH']} \
             SOLVER.OPTIMIZER {hyperparamters['OPTIMIZER']} \
             SOLVER.BETA1 {hyperparamters['BETA1']} \
             SOLVER.BETA2 {hyperparamters['BETA2']} \
             SOLVER.LR_SCHEDULE {hyperparamters['LR_SCHEDULE']} \
             TEST.IMS_PER_BATCH {hyperparamters['TEST_IMS_PER_BATCH']} \
             NHWC True
            """

    print("--------Begin Run Command----------")
    print(run_cmd)
    print("--------End Run Comamnd------------")

    exitcode = 0

    try:
        process = subprocess.Popen(
            run_cmd,
            encoding="utf-8",
            cwd="/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/",
            shell=True,
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
        print(f"mpirun exit code:{exitcode}")
    except Exception as e:
        print("train exception occured", file=sys.stderr)
        exitcode = 1
        print(str(e), file=sys.stderr)

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exitcode)


if __name__ == "__main__":
    train()