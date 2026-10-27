## Installation
### When using HF dataset

```
pip install datasets
```
### When using nemo megatron dataset

```
conda install torchvision torchaudio --override-channels -c pytorch -c conda-forge
pip install Cython
pip install nemo_toolkit['all']
```

## Preparation of datasets
```
sbatch prep/prep_hf_dataset.slurm
```
or
```
sbatch prep/prep_nmt_dataset.slurm
```

## Using prepared datasets
1. Using HF dataset:
You will need to pass at least `--dataset_type hf` and `--training_dir` and `--test_dir` args.

2. Using NMT dataset:
Currently there's a limitation in NMT to only use upto 255 files. That said, refer to the args for `# megatron dataset` in arguments.py.
