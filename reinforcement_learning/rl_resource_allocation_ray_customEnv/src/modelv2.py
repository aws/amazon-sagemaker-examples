import gym
import numpy as np
from gym import spaces
import ray
from pprint import pprint
import ray.rllib.agents.ppo as ppo
from ray.tune import run_experiments
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from ray.rllib.utils.framework import try_import_tf
from bin_packing_env import BinPackingActionMaskGymEnvironment

tf1, tf, tfv = try_import_tf()

class ActionMaskModel(TFModelV2):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kwargs):

        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kwargs)
        
        low = np.asarray(model_config['custom_model_config']['obs_space_low'])
        high = np.asarray(model_config['custom_model_config']['obs_space_high'])
        self.policy = FullyConnectedNetwork(spaces.Box(low, high, shape=(len(low),)),
                                            action_space, num_outputs, model_config, 'policy_network')

    def forward(self, input_dict, state, seq_lens):   
        
        obs = input_dict['obs']['real_obs']
        action_mask = input_dict['obs']['action_mask']
        
        action_logits, _ = self.policy({'obs': obs}, state, seq_lens)

        if self.num_outputs == 1:
            return action_logits, state

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        
        return action_logits + inf_mask, state
    
    def value_function(self):
        return self.policy.value_function()
    
    
def register_actor_mask_model():
    ModelCatalog.register_custom_model("action_mask", ActionMaskModel)