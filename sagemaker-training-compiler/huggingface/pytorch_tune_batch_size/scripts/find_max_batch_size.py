import argparse
import os, subprocess, time
import torch

if __name__ == "__main__":

    os.environ["GPU_NUM_DEVICES"] = str(torch.cuda.device_count())
    
    parser = argparse.ArgumentParser()
    # please update parameters if using a customized training script
    # model configs
    parser.add_argument("--language_modeling_loss", type=str, default="clm", help="select either use training script run_mlm or run_clm")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="HF model name")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="tokenizer name")
    parser.add_argument("--sequence_len", type=str, default="512", help="sequence length")
    parser.add_argument("--fp16", action='store_true', help="whether to train in amp mode")

    # batch size config
    parser.add_argument("--per_device_train_batch_size_min", type=int, default=1, help="minimum batch size to try")
    parser.add_argument("--per_device_train_batch_size_max", type=int, default=256, help="maximum batch size to try")
    args, rem_args = parser.parse_known_args()

    # Causal Language Modeling (run_clm) or Masked Language Modeling (run_mlm)
    training_command = "python "
    if args.language_modeling_loss == "mlm":
        training_command += "./run_mlm.py "
    else:
        training_command += "./run_clm.py "
    training_command += f"--model_name_or_path {args.model_name_or_path} "
    training_command += f"--max_seq_length {args.sequence_len} " if args.language_modeling_loss == "mlm" else (f"--block_size {args.sequence_len} ")
    training_command += "--dataset_name glue "
    training_command += "--dataset_config_name sst2 "
    training_command += "--do_train "
    training_command += "--num_train_epochs 1 "
    training_command += "--max_steps 10 "
    training_command += "--save_strategy no "
    training_command += "--logging_strategy no "
    training_command += "--output_dir /tmp/test "
    training_command += "--overwrite_output_dir "
    if args.fp16:
        training_command += "--fp16 "
    # find max batch size between per_device_train_batch_size_min and per_device_train_batch_size_max
    print("Tuning Command: ", training_command)
    assert args.per_device_train_batch_size_min >= 1 and args.per_device_train_batch_size_min <= args.per_device_train_batch_size_max
    batch_result = 0
    low, high = args.per_device_train_batch_size_min, args.per_device_train_batch_size_max
    tic, i = time.perf_counter(), 0
    while low <= high:
        batch_to_try, i = (low + high) // 2, i + 1
        log_info = f"model: {args.model_name_or_path} trying batch_size: {batch_to_try} "
        try:
            # please update batch_size parameter naming if using a customized training script
            training_command_batch = training_command + f"--per_device_train_batch_size {batch_to_try}"
            subprocess.check_output(training_command_batch, shell=True)
            batch_result, low = batch_to_try, batch_to_try + 1
            log_info += "succeed"
        except subprocess.CalledProcessError as exc:
            high = batch_to_try - 1
            log_info += "failed"
        print(log_info)
    toc = time.perf_counter()

    print(f"Total max batch found in {toc - tic:0.4f} seconds, {i} iterations")
    print(f"[result]: model: {args.model_name_or_path}, max_batch_size between {args.per_device_train_batch_size_min} and {args.per_device_train_batch_size_max} is {batch_result}")
