# Amazon SageMaker Examples

### JumpStart Foundation Models

These examples provide quick walkthroughs to get you up and running with Amazon SageMaker JumpStart Foundation Models.

- [Text2Text Generation Flan T5](text2text-generation-flan-t5.ipynb) demonstrates Text2Text generation using state-of-the-art pretrained Flan T5 models from [Hugging Face](https://huggingface.co/docs/transformers/model_doc/flan-t5) which take an input text containing the task and returns the output of the accomplished task. 
- [Text2Text Generation BloomZ](text2text-generation-bloomz.ipynb) demonstrates Text generation using state-of-the-art pretrained BloomZ 7B1 models from [Hugging Face](https://huggingface.co/bigscience/bloomz-7b1) which take an input text containing the task and returns the output of the accomplished task. In addition to the tasks that Flan T5 can perform, BloomZ can perform multilingual text classification, question and answering, code generation, paragraph rephrase, and More.
- [Text Generation Few Shot Learning](text-generation-few-shot-learning.ipynb) demonstrates Text generation using state-of-the-art pretrained GPT-J-6B models from [Hugging Face](https://huggingface.co/EleutherAI/gpt-j-6B) which takes a text string as input and predicts a sequence of next few words. These models can, for example, fill in incomplete text or paraphrase.
- [Retrieval Augmented Generation: Question and Answering with Embedding](question_answerIng_retrieval_augmented_generation_jumpstart.ipynb) demonstrates how to use retrieval-augmented generation based approach to perform question and answering with JumpStart models.
- [Domain adaption fine-tuning of large language model](domain-adaption-finetuning-gpt-j-6b.ipynb) demonstrates how to use SageMaker SDK to fine-tune large language model like GPT-J-6B from [Hugging Face](https://huggingface.co/EleutherAI/gpt-j-6b) on domain-specific dataset and deploy the fine-tuned model for inference.
