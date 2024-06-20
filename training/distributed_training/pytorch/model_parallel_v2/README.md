## Sagemaker Model Parallelism

This directory contains example scripts to train or fine-tune large scale models,
with the Sagemaker distributed model parallelism library.
When using one of the ipynb notebooks within the folders of this directory please 
make sure to use the `../shared-scripts/` directory as the source directory when submitting a job.

For example, if one wanted to submit a llama finetune job on Sagemaker using the `/llama_v2/smp-finetuning-llama-fsdp-tp.ipynb`
notebook, they would have to upload  `/llama_v2/smp-finetuning-llama-fsdp-tp.ipynb` along with the `../shared-scripts/` directory to Sagemaker notebook.
