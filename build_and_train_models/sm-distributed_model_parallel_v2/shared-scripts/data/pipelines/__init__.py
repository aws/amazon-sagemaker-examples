from data.pipelines.data_pipeline import DataPipeline
from data.pipelines.dummy_data_pipeline import DummyDataPipeline
from data.pipelines.gpt_data_pipeline import GPTDataPipeline
from data.pipelines.hf_data_pipeline import HFDataPipeline


def create_data_pipeline(
    args, start_train_path_index, resume_from_sequence_number, val_resume_from_sequence_number, dp_rank, dp_size
):
    if args.use_synthetic_data:
        data_pipeline = DummyDataPipeline(
            vocabulary_size=args.vocab_size,
            train_batch_size=args.train_batch_size,
            sequence_length=args.max_context_width,
        )
    elif args.dataset_type == "gpt_jsonl":
        data_pipeline = GPTDataPipeline(
            dataset_train_path=args.training_dir,
            train_batch_size=args.train_batch_size,
            dataset_val_path=args.test_dir if args.validation_freq else None,
            val_batch_size=args.val_batch_size if args.validation_freq else None,
            start_path_index=start_train_path_index,
            use_last_file_only_for_valid=args.fast_validation > 0,
            sequence_length=args.max_context_width,
            zipped_data=args.zipped_data,
            seed=args.seed,
            num_workers=args.data_num_workers,
            resume_from_sequence_number=resume_from_sequence_number,
            val_resume_from_sequence_number=val_resume_from_sequence_number,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
    elif args.dataset_type == "hf":
        data_pipeline = HFDataPipeline(
            dataset_train_path=args.training_dir,
            train_batch_size=args.train_batch_size,
            dataset_val_path=args.test_dir if args.validation_freq else None,
            val_batch_size=args.val_batch_size if args.validation_freq else None,
            seed=args.seed,
            num_workers=args.data_num_workers,
            resume_from_sequence_number=resume_from_sequence_number,
            val_resume_from_sequence_number=val_resume_from_sequence_number,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
    return data_pipeline
