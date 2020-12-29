# Download a fraction of Wikipedia corpus for testing BERT distributed training

get a subset of 100 docs 

get the wikiextractor from
attardi/wikiextractor.git (check https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/Dockerfile)

1. Download data
2. clean and format (doc tags are removed from the dataset)
3. sentence segmentation - corpus text file is processed into separate sentens
4. Sharding sentence segmented corpus split into a number of uniformly distributed smaller text docs
5. hdf5 file creation - each text file shard is processed by the `create_pretraining_data.py` to produces a hdf5 file
The scripts generate input data and labels for masked language modeling and sentence prediction tasks for input text shard

The `create_datasets_from_starts.sh` in '/data' applies sentense segmentation, sharding and hdf5 file creation given an arbitrary text file.



Download data
I have downloaded wikipedia abstract in ../donwload.sh

Download google-pretrained-weights, I need the vocab.txt file in it

```
python bertPrep.py --action download --dataset google_pretrained_weights
```
Go checkout stuffs in `~/data/bert/download`



clearn and format data

I have modified the bertPrep so that it takes wikicorpus_en_abstract
I have modfied WikicorpusTextFormatting.py so that it merges abstract into a line
```
python bertPrep.py --action text_formatting --dataset wikicorpus_en_abstract
```

Shard text
Since I ahve created the formatted text, I will just parse the formatted text file to be sharded
The dataset argument below is simply a placeholder
```
python bertPrep.py --action sharding --dataset wikicorpus_en_abstract --input_files /home/ubuntu/data/bert/formatted_one_article_per_line/wikicorpus_en_one_article_per_line.txt \
--n_training_shards 2 --n_test_shards 2
```

In `~/data/bert` I should see a bunch of directories with names `sharded*`, go check them out. 
Shard really just means split dataset into a couple of parts. 

create hdf5
I copied all the .py files from DeepLearningExamples/PyTorch/LanguageModeling/BERT to .
This is because I need the `create_pretraining_data.py` file

checkout the changes I made in the helper function `def create_record_worker`, 

### Create HDF5 files Phase 1

Note the values I parsed to to `n_training_shards` and `n_test_shards` args, 

```
python3 bertPrep.py --action create_hdf5_files --dataset wikicorpus_en_abstract --max_seq_length 128 \
--max_predictions_per_seq 20 --vocab_file /home/ubuntu/data/bert/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1 --n_training_shards 2 --n_test_shards 2
```
Go checkout stuff in `~/data/bert/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5 `


### Create HDF5 files Phase 2
```
python3 bertPrep.py --action create_hdf5_files --dataset wikicorpus_en_abstract --max_seq_length 512 \
--max_predictions_per_seq 80 --vocab_file /home/ubuntu/data/bert/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1 --n_training_shards 2 --n_test_shards 2
```
Go checkout stuff in `hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5`
 



