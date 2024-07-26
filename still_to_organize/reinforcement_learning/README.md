# Amazon SageMaker Examples

### Common Reinforcement Learning Examples

These examples demonstrate how to train reinforcement learning models on SageMaker for a wide range of applications.

**IMPORTANT for rllib users:** Some examples may break with latest [rllib](https://docs.ray.io/en/latest/rllib.html) due to breaking API changes. Please refer to [Amazon SageMaker RL Container](https://github.com/aws/sagemaker-rl-container) for the latest public images and modify the configs in entrypoint scripts according to [rllib algorithm config](https://docs.ray.io/en/latest/rllib-algorithms.html).

If you are using PyTorch rather than TensorFlow, please set `debugger_hook_config=False` when calling `RLEstimator()` to avoid TensorBoard conflicts.

-  [Contextual Bandit with Live Environment](bandits_statlog_vw_customEnv) illustrates how you can manage your own contextual multi-armed bandit workflow on SageMaker using the built-in [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) (VW) container to train and deploy contextual bandit models.
-  [Cartpole](rl_cartpole_coach) uses SageMaker RL base [docker image](https://github.com/aws/sagemaker-rl-container) to balance a broom upright.
-  [Cartpole Spot Training](rl_managed_spot_cartpole_coach) uses SageMaker Managed Spot instances at a lower cost.
-  [Mountain Car](rl_mountain_car_coach_gymEnv) is a classic control RL problem, in which an under-powered car is tasked with climbing a steep mountain, and is only successful when it reaches the top.
-  [Portfolio Management](rl_portfolio_management_coach_customEnv) shows how to re-distribute a capital into a set of different financial assets using RL algorithms.
-  [Predictive Auto-scaling](rl_predictive_autoscaling_coach_customEnv) scales a production service via RL approach by adding and removing resources in reaction to dynamically changing load.
-  [Game Server Auto-pilot](rl_game_server_autopilot) Reduce player wait time by autoscaling game-servers deployed in EKS cluster using RL to add and remove EC2 instances as per dynamic player usage.
-  [Unity Game Agent](rl_unity_ray) shows how to use RL algorithms to train an agent to play Unity3D game.

### FAQ
https://github.com/awslabs/amazon-sagemaker-examples#faq 
