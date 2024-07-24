"""Data pipeline."""
import os
from pathlib import Path

from data.pipelines.data_pipeline import DataPipeline
from logging_utils import get_logger

logger = get_logger()


def make_file_list(dir_path, pattern):
    files = list(Path(dir_path).glob(pattern))
    files = list(set([os.path.join(dir_path, i.stem) for i in files]))
    files.sort()
    files = files[:254]
    proporations = [1 / len(files) for _ in range(len(files))]
    return [val for pair in zip(proporations, files) for val in pair]


# This is still untesed end to end in a convergence run

# Below arguments need to copied to arguments.py to run
# # megatron dataset
# input_grp.add_argument("--data_impl", type=str, default="mmap")
# input_grp.add_argument("--data_split", type=str, default="970, 30, 0")
# input_grp.add_argument("--mmap_warmup", type=int, default=0)
# input_grp.add_argument("--skip_warmup", action="store_true")
# input_grp.add_argument("--tokenizer_type", type=str, default="HFLlamaTokenizer")
# input_grp.add_argument("--tokenizer_vocab_file", type=str, default=None)
# input_grp.add_argument("--tokenizer_merge_file", type=str, default=None)
# input_grp.add_argument("--make_vocab_size_divisible_by", type=int, default=128)
# input_grp.add_argument("--data_dir", type=str)
# input_grp.add_argument("--data_file_regex", type=str)

# Also need to add dataset_type "megatron" as a choice for the arg.

# Below snippet needs to go into data/pipelines/__init__.py
# elif args.dataset_type == "megatron":
#     from data.pipelines.nemo_megatron_gpt_data_pipeline import MegatronGPTDataPipeline

#     data_pipeline = MegatronGPTDataPipeline(
#         args,
#         seed=args.seed,
#         num_workers=args.data_num_workers,
#         resume_from_sequence_number=total_steps,
#         dp_rank=dp_rank,
#         dp_size=dp_size,
#     )


class MegatronGPTDataPipeline(DataPipeline):
    def __init__(
        self,
        args,
        seed=1234,
        num_workers=0,
        resume_from_sequence_number=0,
        dp_rank=0,
        dp_size=1,
        shuffle=False,
    ):
        super().__init__(
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            seed=seed,
            resume_from_sequence_number=resume_from_sequence_number,
            num_workers=num_workers,
            dp_rank=dp_rank,
            dp_size=dp_size,
            shuffle=shuffle,
        )
        eval_iters = (args.max_steps // args.validation_freq + 1) * args.validation_batches

        train_valid_test_num_samples = [
            args.max_steps * args.train_batch_size,
            eval_iters * args.val_batch_size,
            0,
        ]
        logger.info(f"{train_valid_test_num_samples}, {args.max_steps}, {eval_iters}")
        from omegaconf import OmegaConf

        file_list = make_file_list(args.data_dir, args.data_file_regex)
        assert len(file_list) > 0, "Please check your regex"
        model_cfg_dict = {
            "data": {
                # "data_prefix": {
                # "train": make_file_list(args.data_dir, args.data_file_regex),
                # "test": make_file_list(args.data_dir, args.data_file_regex),
                # "validation": make_file_list(args.data_dir, args.data_file_regex),
                # splits_string ignored if data_prefix is a dict
                # },
                "data_prefix": file_list,
                "data_impl": args.data_impl,
                "splits_string": args.data_split,
                "seq_length": args.max_context_width,
                "delay_data_mmap": False,
                "validation_drop_last": True,
                "skip_warmup": args.skip_warmup,
            },
            "seed": args.seed,
        }
        model_cfg = OmegaConf.create(model_cfg_dict)

        from nemo.collections.common.tokenizers import AutoTokenizer

        tokenizer = AutoTokenizer("hf-internal-testing/llama-tokenizer")

        from megatron.core.parallel_state import initialize_model_parallel

        initialize_model_parallel()
        from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import (
            build_train_valid_test_datasets,
        )

        self.train_dataset, self.val_dataset, self.test_dataset = build_train_valid_test_datasets(
            model_cfg,
            None,
            model_cfg.data.data_prefix,
            model_cfg.data.data_impl,
            splits_string=model_cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seq_length=model_cfg.data.seq_length,
            seed=model_cfg.seed,
            skip_warmup=model_cfg.data.get("skip_warmup", True),
            tokenizer=tokenizer,
        )
        self.train_dataloader = self._create_dataloader(self.train_dataset, self.train_batch_size)
        self.val_dataloader = self._create_dataloader(self.val_dataset, self.val_batch_size)
        self.test_dataloader = self._create_dataloader(self.test_dataset, self.val_batch_size)

        logger.info(
            f"Lengths of dataloaders {len(self.train_dataloader)}, {len(self.val_dataloader)}"
        )

    def get_batch(self, data):
        tokens = data["tokens"].long()
        labels = data["labels"].long()
        mask = data["attention_mask"]
        return tokens, mask, labels

    def get_val_batch(self, data):
        tokens = data["tokens"].long()
        mask = data["attention_mask"]
        return tokens, mask
