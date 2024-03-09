import argparse
import json
from functools import partial
from pathlib import Path

from jumpstart_bench.concurrency_probe import num_invocation_scaler
from jumpstart_bench.constants import SAVE_METRICS_FILE_PATH
from jumpstart_bench.payload import create_test_payload
from jumpstart_bench.runner import Benchmarker



def main() -> None:
    args = get_args()

    
    payloads = {
        f"input_{in_tokens:04}_output_{out_tokens:04}": create_test_payload(in_tokens - 1, out_tokens, details=True)
        for in_tokens in args.input_length for out_tokens in args.output_tokens
    }
    num_invocation_hook = partial(num_invocation_scaler, num_invocation_factor=args.num_invocations)
    benchmarker = Benchmarker(
        payloads=payloads,
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
    print(Benchmarker.create_concurrency_probe_pivot_table(
        df,
        value_format_dict={"TokenThroughput": "{:.0f}".format},
        value_name_dict={"TokenThroughput": "throughput (tokens/s)"},
    ))
    print(Benchmarker.create_concurrency_probe_pivot_table(
        df,
        value_format_dict={"LatencyPerToken.Median": "{:.0f}".format},
        value_name_dict={"LatencyPerToken.Median": "median latency (ms/token)"},
    ))
    print(Benchmarker.create_concurrency_probe_pivot_table(
        df,
        value_format_dict={"CostToGenerate1MTokens": "${:,.2f}".format},
        value_name_dict={"CostToGenerate1MTokens": "cost to generate 1M tokens ($)"},
    ))


def get_args() -> argparse.Namespace:
    """Return parsed command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        "-m",
        help="Specify path to a JSON file that contains all model definitions for JumpStart benchmarking.",
    )
    parser.add_argument(
        "--input-length",
        "-i",
        nargs="+",
        type=int,
        default=[256],
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
