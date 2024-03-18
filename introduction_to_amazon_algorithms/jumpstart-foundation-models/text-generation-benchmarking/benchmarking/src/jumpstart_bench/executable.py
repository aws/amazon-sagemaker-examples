import argparse
import json
from functools import partial
import logging
from pathlib import Path

from datasets import Dataset
from jumpstart_bench.concurrency_probe import num_invocation_scaler
from jumpstart_bench.constants import SAVE_METRICS_FILE_PATH
from jumpstart_bench.payload import create_test_payload
from jumpstart_bench.payload import create_test_payload_args
from jumpstart_bench.runner import Benchmarker



def main() -> None:
    args = get_args()
    
    payloads = {
        f"input_{in_tokens:04}_output_{out_tokens:04}": create_test_payload(
            in_tokens - 1, out_tokens, details=True, set_high_temperature=True
        )
        for in_tokens in args.input_length for out_tokens in args.output_tokens
    }

    datasets = {Path(file_name).stem: Dataset.from_json(file_name) for file_name in args.datasets}

    num_invocation_hook = partial(num_invocation_scaler, num_invocation_factor=args.num_invocations)
    benchmarker = Benchmarker(
        payloads=payloads,
        datasets=datasets,
        dataset_payload_keys=create_test_payload_args(args.output_tokens[0], details=True, set_high_temperature=True),
        run_concurrency_probe=True,
        saved_metrics_path=args.metrics_file,
        concurrency_probe_num_invocation_hook=num_invocation_hook,
    )

    if args.skip_run is not True:
        with open(args.models) as f:
            models = json.load(f)

        benchmarker.run_multiple_models(models=models, save_file_path=args.metrics_file)

    if args.skip_clean is not True:
        benchmarker.clean_up_resources()

    df = Benchmarker.load_metrics_pandas(save_file_path=args.metrics_file)
    df_pivot = Benchmarker.create_concurrency_probe_pivot_table(df).reorder_levels(["payload", "instance type", "model ID"]).sort_index()
    
    for metric, df_group in df_pivot.groupby(level=0, axis="columns"):
        df_markdown = df_group.droplevel(0, axis="columns").reset_index().to_markdown()
        logging.info(f"{metric}:\n{df_markdown}\n")


def get_args() -> argparse.Namespace:
    """Return parsed command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        "-m",
        help="Specify path to a JSON file that contains all model definitions for JumpStart benchmarking.",
    )
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        default=[],
        help="Specify the path to any datasets to iterate over during benchmarking."
    )
    parser.add_argument(
        "--input-length",
        "-i",
        nargs="+",
        type=int,
        default=[],
        help="Specify input sequence lengths to benchmark. This can be an arbitrary number, e.g., `-i 256 512`."
    )
    parser.add_argument(
        "--output-tokens",
        "-o",
        nargs="+",
        type=int,
        default=[256],
        help="Specify output sequence lengths to benchmark. This can be an arbitrary number, e.g., `-o 256 512`."
    )
    parser.add_argument(
        "--num-invocations",
        "-n",
        type=int,
        default=3,
        help="Specify the multiplication factor for the number of invocations. Default is 3."
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Flag to skip running benchmarking. Use this to display output of a saved benchmarking file.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Flag to skip cleaning up predictor resources after benchmarking.",
    )
    parser.add_argument(
        "--metrics-file",
        default=SAVE_METRICS_FILE_PATH,
        type=Path,
        help="",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
