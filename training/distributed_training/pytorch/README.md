## Sagemaker Distributed Training

This directory contains example scripts to train or fine-tune large scale models,
with Sagemaker distributed data parallelism and/or model parallelism libraries.
When using one of the ipynb notebooks within the folders of this directory please 
make sure to use the `./shared-scripts/` directory as the source directory when submitting a job.

### Sagemaker Distributed Data Parallelism

Data parallelism examples are in the `./data_parallel/` directory.

### Sagemaker Model Parallelism

Based on the Sagemaker model parallelism (SMP) version, you can find examples scripts in different directories.
- For SMP `v1.x`, please find example scripts in the `./model_parallel/` directory.
   * Each model has its own sub directory and is self-contained.

- For SMP `v2.x`, please find example scripts in the `./model_parallel_v2` directory.
   * **All** models need to access scripts in the `./model_parallel_v2/shared-scripts/` directory.
   * Models have their separate notebooks in sub directories and use the `shared-scripts/` directory as the source directory for their SageMaker jobs.
