

Amazon SageMaker Data Wrangler
=======================================

These example flows demonstrates how to aggregate and prepare data for
Machine Learning using Amazon SageMaker Data Wrangler.


------------------

`Amazon SageMaker Data
Wrangler <https://aws.amazon.com/sagemaker/data-wrangler/>`__ reduces
the time it takes to aggregate and prepare data for ML. From a single
interface in SageMaker Studio, you can import data from Amazon S3,
Amazon Athena, Amazon Redshift, AWS Lake Formation, and Amazon SageMaker
Feature Store, and in just a few clicks SageMaker Data Wrangler will
automatically load, aggregate, and display the raw data. It will then
make conversion recommendations based on the source data, transform the
data into new features, validate the features, and provide
visualizations with recommendations on how to remove common sources of
error such as incorrect labels. Once your data is prepared, you can
build fully automated ML workflows with Amazon SageMaker Pipelines or
import that data into Amazon SageMaker Feature Store.

The `SageMaker example
notebooks <https://sagemaker-examples.readthedocs.io/en/latest/>`__ are
Jupyter notebooks that demonstrate the usage of Amazon SageMaker.

Setup
-------------------------

Amazon SageMaker Data Wrangler is a feature in Amazon SageMaker Studio.
Use this section to learn how to access and get started using Data
Wrangler. Do the following:

-  Complete each step in
   `Prerequisites <https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-getting-started.html#data-wrangler-getting-started-prerequisite>`__.

-  Follow the procedure in `Access Data
   Wrangler <https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-getting-started.html#data-wrangler-getting-started-access>`__
   to start using Data Wrangler.

Examples
-------------------

Tabular Dataflow
---------------------------

.. toctree::
   :maxdepth: 1
   
   tabular-dataflow/index

Timeseries Dataflow
----------------------------

.. toctree::
   :maxdepth: 1
   
   timeseries-dataflow/index

Joined Dataflow
----------------------------

.. toctree::
   :maxdepth: 1
   
   joined-dataflow/index
