export HDF5_USE_FILE_LOCKING=FALSE

python ./DeepLearningExamples/PyTorch/LanguageModeling/BERT/run_pretraining.py \
--input_dir=$SM_CHANNEL_TRAINING \
--output_dir=$SM_MODEL_DIR \
--config_file=./DeepLearningExamples/PyTorch/LanguageModeling/BERT/bert_config.json \
--bert_model=bert-large-uncased \
--train_batch_size=64 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--max_steps=900864 \
--warmup_proportion=0.2843 \
--num_steps_per_checkpoint=900864 \
--learning_rate=1.12e-3 \
--seed=42 \
--fp16 \
--do_train \
--json-summary ./DeepLearningExamples/PyTorch/LanguageModeling/BERT/results/dllogger.json 2>&1

