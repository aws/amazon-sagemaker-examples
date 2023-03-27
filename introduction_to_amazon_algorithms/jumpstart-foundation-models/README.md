# Amazon SageMaker Examples

### JumpStart Foundation Models

These examples provide quick walkthroughs to get you up and running with Amazon SageMaker JumpStart Foundation Models.

- [Text2Text Generation Flan T5](text2text-generation-flan-t5.ipynb) demonstrates Text2Text generation using state-of-the-art pretrained Flan T5 models from [Hugging Face](https://huggingface.co/docs/transformers/model_doc/flan-t5) which take an input text containing the task and returns the output of the accomplished task. 
- [Text2Text Generation BloomZ](text2text-generation-bloomz.ipynb) demonstrates Text generation using state-of-the-art pretrained BloomZ 7B1 models from [Hugging Face](https://huggingface.co/bigscience/bloomz-7b1) which take an input text containing the task and returns the output of the accomplished task. In addition to the tasks that Flan T5 can perform, BloomZ can perform multilingual text classification, question and answering, code generation, paragraph rephrase, and More.
- [Text Generation Few Shot Learning](text-generation-few-shot-learning.ipynb) demonstrates Text generation using state-of-the-art pretrained GPT-J-6B models from [Hugging Face](https://huggingface.co/EleutherAI/gpt-j-6B) which takes a text string as input and predicts a sequence of next few words. These models can, for example, fill in incomplete text or paraphrase.
- [Retrieval Augmented Generation: Question and Answering with Embedding](retrieval_augmented_generation_question_answering_embedding.ipynb) demonstrates how to use retrieval-augmented generation based approach to perform question and answering with JumpStart models.