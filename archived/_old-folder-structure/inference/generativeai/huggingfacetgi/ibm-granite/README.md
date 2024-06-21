# ibm-granite



## Introduction

As enterprises modernize their mission-critical applications to adopt cloud-native architectures and containerized microservices, a major challenge is converting legacy monolithic codebases to modern languages and frameworks. Manual code conversion is extremely time-consuming, expensive, and error-prone. Fortunately, recent advances in large language models (LLMs) for code have opened up new possibilities for AI-assisted code conversion at scale.

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a leading machine learning platform that makes it easy to build, train, and deploy machine learning models in the cloud and at the edge. IBM has recently open-sourced its powerful [Granite](https://github.com/ibm-granite/granite-code-models) family of code LLMs that excel at code generation, translation, fixing bugs, and more across over 100 programming languages. By combining the strengths of SageMaker and Granite Code models, enterprises can now accelerate legacy code conversion projects.

In this [notebook](granite-code-instruct.ipynb), you will learn how to deploy IBM Granite models on Amazon SageMaker for accelerating legacy code conversion and modernisation use cases.


## What are Granite models

The IBM Granite Code models are a family of high-performance, foundational language models pre-trained on over 3 trillion tokens of code and natural language data across 116 programming languages. These models range from 3 billion to 34 billion parameters and come in base and instruction-following variants.

IBM has released the Granite Code models to open source under the permissive Apache 2.0 license, enabling their use for both research and commercial purposes with no restrictions. The models are available on [Hugging Face](https://huggingface.co/ibm-granite).


This notebook shows how to deploy ibm-granite/granite-20b-code-instruct, an open foundation model for code intelligence, to an Amazon SageMaker real-time endpoint with TGI backend.

## What are Granite models
- [https://github.com/ibm-granite/granite-code-models](https://github.com/ibm-granite/granite-code-models)
- [https://huggingface.co/ibm-granite](https://huggingface.co/ibm-granite)
- [https://github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)
