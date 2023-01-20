FROM nvcr.io/nvidia/pytorch:22.10-py3

RUN pip install sagemaker transformers==4.9.1 tritonclient[all]

RUN pip install ipykernel && python -m ipykernel install --sys-prefix