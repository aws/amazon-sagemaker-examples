### SageMaker Image classification full training
This notebook `ImageClassification-fulltraining.ipynb` demos an end-2-end system for image classification training using resnet model. Caltech-256 dataset is used as a sample dataset. Various parameters such as network depth (number of layers), batch_size, learning_rate, etc., can be varied in the training. Once the training is complete, the notebook shows how to host the trained model for inference.

### SageMaker Image classification transfer learning
This notebook `Imageclassification-transfer-learning.ipynb` demos an end-2-end system for image classification fine-tuning using a pre-trained resnet model on imagenet dataset. Caltech-256 dataset is used as a transfer learning dataset. The network re-initializes the output layer with the number of classes in the Caltech dataset and retrains the layer while at the same time fine-tuning the other layers. Various parameters such as network depth (number of layers), batch_size, learning_rate, etc., can be varied in the training. Once the training is complete, the notebook shows how to host the trained model for inference.

### SageMaker Image classification lst format
This notebook `Imageclassification-lst-format.ipynb` demos an end-2-end system for image classification training with image and list files. Caltech-256 dataset is used as a transfer learning dataset. The network re-initializes the output layer with the number of classes in the Caltech dataset and retrains the layer while at the same time fine-tuning the other layers. Various parameters such as network depth (number of layers), batch_size, learning_rate, etc., can be varied in the training. Once the training is complete, the notebook shows how to host the trained model for inference.

### SageMaker Image classification full training highlevel
This notebook `ImageClassification-fulltraining-highlevel.ipynb` is similar to the `ImageClassification-fulltraining.ipynb` but using Sagemaker high-level APIs

### SageMaker Image classification transfer learning highlevel
This notebook `Imageclassification-transfer-learning-highlevel.ipynb` is similar to the `ImageClassification-transfer-learning.ipynb` but using Sagemaker high-level APIs

### SageMaker Image classification lst format highlevel
This notebook `Imageclassification-lst-format-highlevel.ipynb` is similar to the `ImageClassification-lst-format.ipynb` but using Sagemaker high-level APIs
