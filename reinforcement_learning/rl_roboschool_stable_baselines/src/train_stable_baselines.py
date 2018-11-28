import argparse

from sagemaker_rl.mpi_launcher import MPILauncher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--RLSTABLEBASELINES_PRESET', required=True, type=str)
    parser.add_argument('--output_path', default="/opt/ml/output/intermediate/", type=str)
    parser.add_argument('--instance_type', type=str)

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown_args = parse_args()
    print("Launching train script with MPI: {} and arguments: {}".format(args.RLSTABLEBASELINES_PRESET,
                                                                         str(unknown_args)))
    MPILauncher(train_script=args.RLSTABLEBASELINES_PRESET, train_script_args=unknown_args,
                instance_type=args.instance_type).mpi_run()
