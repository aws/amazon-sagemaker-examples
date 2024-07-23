# Amazon SageMaker Faster Autoscaling

To demonstrate  newer, faster SageMaker autoscaling features, We deploy Meta's **Llama3-8B-Instruct** model to an Amazon SageMaker real-time endpoint using Text Generation Inference (TGI) Deep Learning Container (DLC).

To trigger autoscaling, we need to generate traffic to the endpoint.
We use [LLMPerf](https://github.com/philschmid/llmperf) to generate sample traffic to the endpoint.

## Prerequisites

Before using this notebook please ensure you have access to an active access token from HuggingFace and have accepted the license agreement from Meta.

- Step 1: Create user access token in HuggingFace (HF). Refer [here](https://huggingface.co/docs/hub/security-tokens) on how to create HF tokens.
- Step 2: Login to HuggingFace and navigate to [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/tree/main) home page.
- Step 3: Accept META LLAMA 3 COMMUNITY LICENSE AGREEMENT by following the instructions [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/tree/main).
- Step 4: Wait for the approval email from META (Approval may take any where b/w 1-3 hrs).

---

>NOTE: LLMPerf spins up a ray cluster to generate traffic to Amazon SageMaker endpoint.\
>When running this on Amazon SageMaker Notebook Instance, ensure you use at least **m5.2xlarge** or a larger instance type.

## Autoscaling on real-time endpoints

### Amazon SageMaker real-time endpoints

- For Application Autoscaling example on Amazon SageMaker real-time endpoints refer to [FasterAutoscaling-SME-Llama3-8B-AppAutoScaling.ipynb](./realtime-endpoints/FasterAutoscaling-SME-Llama3-8B-AppAutoScaling.ipynb) notebook.

- For StepScaling example on Amazon SageMaker real-time endpoints refer to [FasterAutoscaling-SME-Llama3-8B-StepScaling.ipynb](./realtime-endpoints/FasterAutoscaling-SME-Llama3-8B-StepScaling.ipynb) notebook.

### Amazon SageMaker Inference Components

- For autoscaling example using Amazon SageMaker Inference components, refer to [inference-component-llama3-autoscaling.ipynb](./realtime-endpoints/FasterAutoscaling-IC-Llama3-8B-AppAutoScaling.ipynb) notebook.

---

## References

- [LLMPerf](https://github.com/philschmid/llmperf)
- [Llama3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [Create HF Access Token](https://huggingface.co/docs/hub/security-tokens)
- [Amazon SageMaker Inference Components - blog post](https://aws.amazon.com/blogs/machine-learning/reduce-model-deployment-costs-by-50-on-average-using-sagemakers-latest-features/)