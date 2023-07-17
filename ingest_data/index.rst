
.. image:: image/data_ingestion_header.png
  :width: 600
  :alt: data ingestion

Get started with data ingestion
===============================

You have several different options for how can access your data from SageMaker.
The most commonly used data source in the examples uses S3 buckets.
You can also use Athena, EMR, Redshift, and Amazon Keyspaces DB as data sources.


Basic S3 examples by data type
------------------------------
The following notebooks demonstrate how to load the most popular data types: images, tabular, and text.
Note that you don't have to have an S3 bucket to get started.
SageMaker uses a `default bucket <https://sagemaker.readthedocs.io/en/stable/api/utility/session.html?highlight=default%20bucket#sagemaker.session.Session.default_bucket>`_ that it creates for you as an easy way to get started without having to create one yourself.

.. toctree::
   :maxdepth: 1

   ingest-data-types/ingest_tabular_data


Athena
=============

You can use Amazon Athena as a data source for SageMaker.
Athena is a serverless interactive query service that makes it easy to analyze your S3 data with standard SQL.
This example runs the California housing dataset and uses PyAthena, a Python client for Athena, and `awswrangler`, a Pandas-like interface to many AWS data platforms.

.. toctree::
   :maxdepth: 1

   ingest-with-aws-services/ingest_data_with_Athena




Redshift
==================

You can use Amazon Redshift as a data source for SageMaker.
Redshift is a fully managed data warehouse that allows you to run complex analytic queries against petabytes of structured data.
This example runs the California housing dataset and uses `awswrangler`, a Pandas-like interface to many AWS data platforms.


.. toctree::
   :maxdepth: 1

   ingest-with-aws-services/ingest_data_with_Redshift


