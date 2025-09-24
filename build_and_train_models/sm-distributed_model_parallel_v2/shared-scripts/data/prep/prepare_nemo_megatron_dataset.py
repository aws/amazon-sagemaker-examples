import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SRC_DIR = "/fsx/datasets/c4/en/hf/"
OUT_DIR = "/fsx/datasets/c4/en/nmt-tokenized-2/llama"

if not Path(OUT_DIR).exists():
    os.makedirs(OUT_DIR)


def process_file(idx):
    file_idx_str = str(idx).zfill(5)
    file_stem = f"c4-train.{file_idx_str}-of-01024"
    file_name = f"{file_stem}.json.gz"
    cmd = f"python data/_prepare_nemo_megatron_dataset.py \
                --input {os.path.join(SRC_DIR, file_name)} \
                --output-prefix {OUT_DIR}/{file_stem} \
                --tokenizer-library=huggingface \
                --tokenizer-type hf-internal-testing/llama-tokenizer \
                --dataset-impl mmap \
                --append-eod \
                --workers 32"
    os.system(cmd)
    output_partition_files = list(Path(OUT_DIR).glob(f"{file_stem}_[0-9]*"))
    # Running with 2 partitions creates some extra files we don't need
    for a_file in output_partition_files:
        a_file.unlink()
    input_partition_files = list(Path(SRC_DIR).glob(f"{file_stem}.json_[0-9].gz"))
    for a_file in input_partition_files:
        a_file.unlink()


pool = ThreadPoolExecutor(max_workers=32)

# import os
# node_id = int(os.getenv('SLURM_NODEID'))
# num_nodes = int(os.getenv('SLURM_NNODES'))
threads = [pool.submit(process_file, idx) for idx in range(95, 256)]
