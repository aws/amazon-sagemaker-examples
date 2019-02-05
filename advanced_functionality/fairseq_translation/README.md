# Fairseq on Amazon SageMaker

[Fairseq](https://github.com/pytorch/fairseq) is a sequence modeling toolkit created by Facebook AI Research. It allows to train and serve custom models for translation, summarization, language modeling and other text generation tasks. It also provides [reference implementations](https://github.com/pytorch/fairseq#introduction-) of various sequence-to-sequence models. 

In the following examples, we will show how to integrate Fairseq into Amazon SageMaker by creating your own container and using it to train and serve predictions. 

## Example notebooks
* `fairseq_sagemaker_translate_de2en.ipynb`: end-to-end training example of training a German-English translation model
* `fairseq_sagemaker_distributed_translate_de2en.ipynb`: end-to-end **multi-machine** example of training a German-English translation model 
* `fairseq_sagemaker_pretrained_en2fr.ipynb`: example of using a pre-trained English-French model to serve predictions and test the inference experience
* `fairseq_sagemaker_translate_en2fr.ipynb`: end-to-end example of training an English-French translation model

## Supported version
The examples are using Fairseq [v0.6.0](https://github.com/pytorch/fairseq/releases/tag/v0.6.0). 