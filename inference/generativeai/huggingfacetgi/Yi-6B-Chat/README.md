#  Hosting Yi-6B-Chat on Amazon SageMaker using HuggingFace Text Generation Inference (TGI)

This notebook demonstrates how to deploy Hosting Yi-6B-Chat transformer models using Hugging Face Text Generation Inference (TGI) Deep Learning Container on Amazon SageMaker.

The [Yi series](https://huggingface.co/01-ai) models are the next generation of open-source large language models trained from scratch by [01.AI](https://01.ai/).For English language capability, the Yi series models ranked 2nd (just behind GPT-4), outperforming other LLMs (such as LLaMA2-chat-70B, Claude 2, and ChatGPT) on the [AlpacaEval Leaderboard](https://gitlab.aws.dev/arraafat/yi-6b-chat-on-amazon-sagemaker/-/edit/master/README.md?ref_type=heads) in Dec 2023.

 ![AlpacaEval Leaderboard](/YI.png)
TGI is an open source, high performance inference library that can be used to deploy large language models from Hugging Face’s repository in minutes. The library includes advanced functionality like model parallelism and dynamic batching to simplify production inference with large language models like flan-t5-xxl, LLaMa, StableLM, and GPT-NeoX.
