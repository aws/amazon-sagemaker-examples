# Train Machine Learning Models using Amazon Keyspaces as a Data Source  

In this notebook we will provide step-by-step instruction to use SageMaker to ingest customer data from Amazon Keyspaces and train a clustering model that allowed you to segment customers. You could use this information for targeted marketing, greatly improving your business KPI.

1. First, we install Sigv4 driver to connect to Amazon Keyspaces

> The Amazon Keyspaces SigV4 authentication plugin for Cassandra client drivers enables you to authenticate calls to Amazon Keyspaces ***using IAM access keys instead of user name and password***. To learn more about how the Amazon Keyspaces SigV4 plugin enables [`IAM users, roles, and federated identities`](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) to authenticate in Amazon Keyspaces API requests, see [`AWS Signature Version 4 process (SigV4)`](https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html)

2. Next, we establish a connection to Amazon Keyspaces 
3. Next, we create new Keyspace ***blog_(yyyymmdd)*** and a new table ***online_retail*** 
3. Next, we will download retail data about customers.
3. Next, we will ingest retail data about customers into Keyspaces.
3. Next, we use a notebook available within SageMaker Studio to collect data from Keyspaces database, and prepare data for training using KNN Algorithm. Most of our customers use SageMaker Studio for end to end development of ML Use Cases. They could use this notebook as a base and customize it quickly for their use case. Additionally, they will be able to share this with other collaborators without requiring them to install any additional software. 
3. Next, we will train the data for clustering.
3. After the training is complete, we can view the mapping between customer and their associated cluster.
3. And finally, Cleanup Step to drop Keyspaces table to avoid future charges. 

Contributers 
- `Vadim Lyakhovich (AWS)`
- `Ram Pathangi (AWS)`
- `Parth Patel (AWS)`

### Note
The notebook execution role must include permissions to access Amazon Keyspaces and Assume the role.

*  To access Amazon Keyspaces database - use `AmazonKeyspacesReadOnlyAccess` or `AmazonKeyspacesFullAccess` managed policies. Use the _least privileged approach_ for your production application.  
See more at
[`AWS Identity and Access Management for Amazon Keyspaces`](https://docs.aws.amazon.com/keyspaces/latest/devguide/security-iam.html).

* To assume the role, you need to have [`sts:AssumeRole action`](https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html) permissions.
    ```bash
    {
      "Version": "2012-10-17",  
      "Statement": [  
        {  
           "Action": [  
           "sts:AssumeRole"  
          ],  
          "Effect": "Allow",  
          "Resource": "*"  
        }
      ]
    }
    ```

This notebook was tested with `conda_python3` kernel and should work with `Python 3.x`.


*Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved*  
[*SPDX-License-Identifier: MIT-0*](https://github.com/aws/mit-0)
