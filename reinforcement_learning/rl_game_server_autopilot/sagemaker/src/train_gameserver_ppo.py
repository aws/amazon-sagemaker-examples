import json
import os
import sys
import gym
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

from sagemaker_rl.ray_launcher import SageMakerRayLauncher

env_config={}

class MyLauncher(SageMakerRayLauncher):

    def register_env_creator(self):
        from gameserver_env import GameServerEnv
        register_env("GameServers", lambda env_config: GameServerEnv(env_config))
        
    def _save_tf_model(self):
        print("in _save_tf_model")
        ckpt_dir = '/opt/ml/output/data/checkpoint'
        model_dir = '/opt/ml/model'

        # Re-Initialize from the checkpoint so that you will have the latest models up.
        tf.train.init_from_checkpoint(ckpt_dir,
                                      {'main_level/agent/online/network_0/': 'main_level/agent/online/network_0'})
        tf.train.init_from_checkpoint(ckpt_dir,
                                      {'main_level/agent/online/network_1/': 'main_level/agent/online/network_1'})

        # Create a new session with a new tf graph.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(tf.global_variables_initializer())  # initialize the checkpoint.

        # This is the node that will accept the input.
        input_nodes = tf.get_default_graph().get_tensor_by_name('main_level/agent/main/online/' + \
                                                                'network_0/observation/observation:0')
        # This is the node that will produce the output.
        output_nodes = tf.get_default_graph().get_operation_by_name('main_level/agent/main/online/' + \
                                                                    'network_1/ppo_head_0/policy_mean/BiasAdd')
        # Save the model as a servable model.
        tf.saved_model.simple_save(session=sess,
                                   export_dir='model',
                                   inputs={"observation": input_nodes},
                                   outputs={"policy": output_nodes.outputs[0]})
        # Move to the appropriate folder. 
        shutil.move('model/', model_dir + '/model/tf-model/00000001/')
        # SageMaker will pick it up and upload to the right path.
        print("in _save_tf_model Success")

    def get_experiment_config(self):
        print('get_experiment_config')       
        print(env_config)
        # allowing 1600 seconds to the job toto stop and save the model
        time_total_s=float(env_config["time_total_s"])-4600
        print("time_total_s="+str(time_total_s))
        return {
          "training": {
            "env": "GameServers",
            "run": "PPO",
             "stop": {
               "time_total_s": time_total_s
             },
            "config": {
               "ignore_worker_failures": True,
               "gamma": 0,
               "kl_coeff": 1.0,
               "num_sgd_iter": 10,
               "lr": 0.0001,
               "sgd_minibatch_size": 32, 
               "train_batch_size": 128,
               "model": {
#                 "free_log_std": True,
#                  "fcnet_hiddens": [512, 512],
                },
               "use_gae": True,
               #"num_workers": (self.num_cpus-1),
               "num_gpus": self.num_gpus,
               #"batch_mode": "complete_episodes",
               "num_workers": 1,
                "env_config": env_config,
               #'observation_filter': 'MeanStdFilter',
            }
          }
        }

if __name__ == "__main__":
    for i in range(len(sys.argv)):
      if i==0:
         continue
      if i % 2 > 0:
         env_config[sys.argv[i].split('--',1)[1]]=sys.argv[i+1]
    MyLauncher().train_main()
