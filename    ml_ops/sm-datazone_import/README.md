# DataZone Import SageMaker Domain

This example contains python scripts to import an existing SageMaker Domain into DataZone. Intended to be ran by administrators.

## Setup

1. Add the Bring-Your-Own-Domain (BYOD) service model

```bash
aws configure add-model --service-model file://resources/datazone-linkedtypes-2018-05-10.normal.json --service-name datazone-byod
```

2. Create a federation role

This role will be used by DataZone to launch the SageMaker Domain. See [BringYourOwnDomainResources.yml](.resources/BringYourOwnDomainResources.yml) for an example.

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
    --federation-role ARN_OF_FEDERATION_ROLE \
    --account-id ACCOUNTID
```

### Additional Configuration

- SageMaker execution roles need DataZone API permissions in order for the Assets UI to function. See [DataZoneUserPolicy.json](./resources/DataZoneUserPolicy.json) for an example.
- Ensure the DataZone Domain trusts SageMaker. In the AWS DataZone console navigate to Domain details and select the "Trusted services".