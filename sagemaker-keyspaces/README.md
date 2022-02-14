# Train Machine Learning Models using Amazon Keyspaces as a Data Source.  

In this lab we will provide step-by-step instruction to use SageMaker to ingest customer data from Amazon Keyspaces and train a clustering model that allowed you to segment customers. You could use this information for targeted marketing, greatly improving your business KPI.

1. First, we install Sigv4 driver to connect to Amazon Keyspaces 

> The Amazon Keyspaces SigV4 authentication plugin for Cassandra client drivers enables you to authenticate calls to Amazon Keyspaces ***using IAM access keys instead of user name and password***. To learn more about how the Amazon Keyspaces SigV4 plugin enables [IAM users, roles, and federated identities](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) to authenticate in Amazon Keyspaces API requests, see [AWS Signature Version 4 process (SigV4)](https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html)

2. Next, we establish a connection to Amazon Keyspaces 
3. Next, we create new Keyspace ***blog*** and a new table ***online_retail*** 
3. Next, we will download retail data about customers.
3. Next, we will ingest retail data about customers into Keyspaces.
3. Next, we will read the ingested data into Sagemaker and do feature engineering.
3. Next, we will train the data for clustering.
3. After the training is complete, we can view the mapping between customer and their associated cluster.
3. And finally, Cleanup step to drop Keyspaces table to avoid future charges. 

Created by 
- Vadim Lyakhovich (AWS)
- Ram Pathangi (AWS)
- Parth Patel (AWS)



*Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved*  
[*SPDX-License-Identifier: MIT-0*](https://github.com/aws/mit-0)
