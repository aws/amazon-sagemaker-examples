.. image:: https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png
    :height: 100px
    :alt: SageMaker

====================
SageMaker Core
====================

.. image:: https://img.shields.io/pypi/v/sagemaker-core.svg
   :target: https://pypi.python.org/pypi/sagemaker-core
   :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/sagemaker-core.svg
   :target: https://pypi.python.org/pypi/sagemaker-core
   :alt: Supported Python Versions



Introduction
------------

Welcome to the sagemaker-core Python SDK, an SDK designed to provide an object-oriented interface for interacting with Amazon SageMaker resources. It offers full parity with SageMaker APIs, allowing developers to leverage all SageMaker capabilities directly through the SDK. sagemaker-core introduces features such as dedicated resource classes, resource chaining, auto code completion, comprehensive documentation and type hints to enhance the developer experience as well as productivity. 


Key Features
------------

* **Object-Oriented Interface**: Provides a structured way to interact with SageMaker resources, making it easier to manage them using familiar object-oriented programming techniques.
* **Resource Chaining**: Allows seamless connection of SageMaker resources by passing outputs as inputs between them, simplifying workflows and reducing the complexity of parameter management.
* **Full Parity with SageMaker APIs**: Ensures access to all SageMaker capabilities through the SDK, providing a comprehensive toolset for building and deploying machine learning models.
* **Abstraction of Low-Level Details**: Automatically handles resource state transitions and polling logic, freeing developers from managing these intricacies and allowing them to focus on higher-level tasks.
* **Auto Code Completion**: Enhances the developer experience by offering real-time suggestions and completions in popular IDEs, reducing syntax errors and speeding up the coding process.
* **Comprehensive Documentation and Type Hints**: Provides detailed guidance and type hints to help developers understand functionalities, write code faster, and reduce errors without complex API navigation.
* **Incorporation of Intelligent Defaults**: Integrates the previous SageMaker SDK feature of intelligent defaults, allowing developers to set default values for parameters like IAM roles and VPC configurations. This streamlines the setup process, enabling developers to focus on customizations specific to their use case.


Benefits
--------

* **Simplified Development**: By abstracting low-level details and providing intelligent defaults, developers can focus on building and deploying machine learning models without getting bogged down by repetitive tasks.
* **Increased Productivity**: The SDK's features, such as auto code completion and type hints, help developers write code faster and with fewer errors.
* **Enhanced Readability**: Resource chaining and dedicated resource classes result in more readable and maintainable code.


Docs and Examples
-----------------
Learn more about the sagemaker-core SDK and its features by visting the `What's New Announcement <https://aws.amazon.com/about-aws/whats-new/2024/09/sagemaker-core-object-oriented-sdk-amazon-sagemaker>`_.

For examples and walkthroughs, see the `SageMaker Core Examples <https://github.com/aws/amazon-sagemaker-examples/tree/default/%20%20%20%20%20%20%20%20%20sagemaker-core>`_.

For detailed documentation, including the API reference, see `Read the Docs <https://sagemaker-core.readthedocs.io>`_.
