# README

Builds a lambda layer with a newer version of boto3, this is required
for starting "streaming" labeling jobs.

We don't commit all of boto3 + dependencies in order to prevent cluttering the
source repository.

## Build

This command will install all the layer's dependencies to "target".
```
./build.sh
```

## Deploy
This layer, once built, is deployed using terraform.
