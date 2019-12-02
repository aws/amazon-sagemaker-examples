## Amazon SageMaker Debugger Examples

This repository contains example notebooks that show how to use Amazon SageMaker Debugger. It's a feature in SageMaker that allows you to look inside the training process and debug it by analyzing data emitted. Note that although these notebooks focus on a specific framework, the same approach works with all the frameworks that Amazon SageMaker Debugger supports. These are in the order in which we recommend you review them.

- [Using a built-in rule with TensorFlow](using_rules/tf-mnist-builtin-rule.ipynb) 
- [Using a custom rule with TensorFlow](using_rules/tf-mnist-custom-rule.ipynb)
- [Using Cloudwatch events from rules to stop training jobs if rule condition is met](using_rules/tf-mnist-stop-training-job.ipynb)
- [Interactive tensor analysis in notebook with MXNet](mnist_tensor_analysis/)
- [Real time analysis in notebook with MXNet](mxnet_realtime_analysis/)
