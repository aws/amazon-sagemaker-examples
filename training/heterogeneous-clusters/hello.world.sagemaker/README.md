# Hetero Training Job - Hello world
This basic example on how to run a Heterogeneous Clusters training job consisting of two instance groups. Each instance group includes a different instance type.  
Each instance prints its environmental information including its instance group and exits.
This demo doesn't include applying a distribution to one of the instance groups (e.g., for distributed training)

Environment information can be obtained in two ways:
  - `Option-1`: Read instance group information using the convinient sagemaker_training.environment.Environment class.
  - `Option-2`: Read instance group information from `/opt/ml/input/config/resourceconfig.json`.

## Running the example:
Start a SageMaker training job:
```bash
cd ./hello.world.sagemaker/
python3 ./start_job.py
```
Wait for the training job to finish and review its logs in the AWS Console. You'll find two logs: Algo1, Algo2. Examine the printouts on each node on how to retrieve instance group environment infomation.

Next, See the TensorFlow or PyTorch examples.