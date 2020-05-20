# Game Server Autopilot with Amazon SageMaker RL

Multiplayer game publishers often need to either over-provision resources or manually manage compute resource allocation when launching a large-scale worldwide game, to avoid the long player-wait in the game lobby. Game publishers need to develop, config, and deploy tools that helped them to monitor and control the compute allocation. This blog demonstrates GameServer Autopilot, a new machine learning-based example tool that makes it easy for game publishers to reduce the time players wait for compute to spawn, while still avoiding compute over-provisioning. It also eliminates manual configuration decisions and changes publishers need to make and reduces the opportunity for human errors.

Here we describe a reinforcement learning-based system that learns to allocate resources in response to player usage patterns. The hosted model directly predicts the required number of game-servers so as to allow EKS the time to allocate instances to reduce player wait time. The training process integrates with the game eco-system, and requires minimal manual configuration.

## Contents

### SageMaker

  * `rl_gamerserver_ray.ipynb`: notebook used for training predictive game server auto-scaling policy.
  * `src/`
     * `gameserver_env.py`: custom game server environments and simulator defined here.
     * `train-gameserver_ppo.py`: launcher for ppo training.
     * `evaluate-gameserver.py`: launcher for coach evaluation.

### Game Environment (EKS, DynamoDB)

  * `workspace_prep.sh`: cloud 9 init script
  * `create_aws_objects.sh`: create DynamodBD table, ECR
  * EKS deployment scripts
  * `minecraft-server-image`: sample game server

### Autopilot Server

  * `app.py`: Chalice-based http server
  * `requirements.txt`: python packages needed

### Autopilot Client

  * `autopilot-client-image`: Docker images script for ap-client
  * `specs/autopilot-client.yaml`: EKS spec for ap-client
