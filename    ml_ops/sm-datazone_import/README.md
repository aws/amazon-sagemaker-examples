# DataZone Import SageMaker Domain

This example contains python scripts to import an existing SageMaker Domain into DataZone. Intended to be ran by administrators.

## Setup

1. Add the Bring-Your-Own-Domain (BYOD) service model

```bash
aws configure add-model --service-model file://resources/datazone-linkedtypes-2018-05-10.normal.json --service-name datazone-byod
```

2. Create a federation role

This role will be used by DataZone to launch the SageMaker Domain. See [BringYourOwnDomainResources.yml](.resources/BringYourOwnDomainResources.yml) for an example.
If you are using multiple projects with multiple users, you will need to create a separate SageMaker Execution Role as well as Federation Role.

### Prerequisites

- SageMaker Domain with users.
- DataZone Domain and Project with users.
- Have a good idea on which SageMaker users should map to which DataZone users.

#### Limitations

- DataZone Domain and SageMaker Domain must be in the same region.

### Running the script

Run the script and follow the instructions.

```bash
python import-sagemaker-domain.py \
    --region REGION \
    --account-id ACCOUNTID
```

### Additional Configuration

- SageMaker execution roles need DataZone API permissions in order for the Assets UI to function. See [DataZoneUserPolicy.json](./resources/DataZoneUserPolicy.json) for an example.
- Ensure the DataZone Domain trusts SageMaker. In the AWS DataZone console navigate to Domain details and select the "Trusted services".

### Potential errors and workarounds

**Cannot view ML assets in SageMaker Studio**

Make sure that the execution role that is attached to the SageMaker User in the attached domain has ListTags attached as a permissions policy to the role. A simple workaround is to attach AmazonSageMakerCanvasFullAccess policy which contains this permission. Without it - you will not be able to view the Assets tab in the Studio UI. Expected error in the UI inspect: 
```
User: arn:aws:sts::789706018617:assumed-role/AmazonSageMaker-ExecutionRole-20241127T120959/SageMaker is not authorized to 
perform: sagemaker:ListTags on resource: arn:aws:sagemaker:us-east-1:789706018617:domain/d-qy9jzu4s7q0y because no 
identity-based policy allows the sagemaker:ListTags action
```

**Able to view assets in sidebar, but page is not loading.**

If you are able to view the assets - but are getting a There was a problem when loading subscriptions error in the page where your ML assets should be - ensure that the SageMaker Execution role tied to this SageMaker user has AmazonDataZoneFullAccess or a more limited AmazonDataZoneFullUserAccess attached to it. 

**DataZone portal is not showing a generated action-link for user.**

If you are attempting to create ProjectB using a subset of users B under created environment B - make sure. that you use a separate federation role when the _associate_fed_role action is called. This is required or else the association will fail and thus the subsequent call to create_environment_action will fail with the following error. 
See `../resources` for sample permissions and trust policies for the federation role. Be sure to fill in your SageMaker Domain Id.

```
An error occurred (ValidationException) when calling the AssociateEnvironmentRole operation: Role Arn 
arn:aws:iam::789706018617:role/svia-test-byod-fed-role already being used in a different project
```

Successful association will return the following

```
Federation role to federate into sagemaker studio from datazone portal: arn:aws:iam::789706018617:role/svia-test-byod-fed-role
Associating Environment Role using Federation Role [arn:aws:iam::789706018617:role/svia-test-byod-fed-role] ...
Associating Environment Role using Federation Role [arn:aws:iam::789706018617:role/svia-test-byod-fed-role] COMPLETE
```