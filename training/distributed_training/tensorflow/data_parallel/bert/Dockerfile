ARG region

FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.3.1-gpu-py37-cu110-ubuntu18.04

RUN 	pip --no-cache-dir --no-cache install \
        scikit-learn==0.23.1 \
        wandb==0.9.1 \
        tensorflow-addons \
        colorama==0.4.3 \
        pandas \
        apache_beam \
        pyarrow==0.16 \
        git+https://github.com/HerringForks/transformers.git@master \
        git+https://github.com/huggingface/nlp.git@703b761
        