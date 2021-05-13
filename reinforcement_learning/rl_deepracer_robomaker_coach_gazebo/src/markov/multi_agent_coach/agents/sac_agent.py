#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import copy
from collections import OrderedDict
from typing import Union

import numpy as np
import tensorflow as tf
from markov.multi_agent_coach.architectures.head_parameters import (
    SACPolicyHeadParameters,
    SACQHeadParameters,
)
from rl_coach.agents.agent import Agent
from rl_coach.agents.policy_optimization_agent import PolicyOptimizationAgent
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import VHeadParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.architectures.network_wrapper import NetworkWrapper
from rl_coach.base_parameters import (
    AgentParameters,
    AlgorithmParameters,
    EmbedderScheme,
    MiddlewareScheme,
    NetworkParameters,
)
from rl_coach.core_types import ActionInfo, EnvironmentSteps, RunPhase
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplayParameters
from rl_coach.spaces import BoxActionSpace

# There are 3 networks in SAC implementation. All have the same topology but parameters are not shared.
# The networks are:
# 1. State Value Network - SACValueNetwork
# 2. Soft Q Value Network - SACCriticNetwork
# 3. Policy Network - SACPolicyNetwork - currently supporting only Gaussian Policy


# 1. State Value Network - SACValueNetwork
# this is the state value network in SAC.
# The network is trained to predict (regression) the state value in the max-entropy settings
# The objective to be minimized is given in equation (5) in the paper:
#
# J(psi)= E_(s~D)[0.5*(V_psi(s)-y(s))^2]
# where y(s) = E_(a~pi)[Q_theta(s,a)-log(pi(a|s))]


# Default parameters for value network:
# topology :
#   input embedder : EmbedderScheme.Medium (Dense(256)) , relu activation
#   middleware : EmbedderScheme.Medium (Dense(256)) , relu activation


class SACValueNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {
            "observation": InputEmbedderParameters(activation_function="relu")
        }
        self.middleware_parameters = FCMiddlewareParameters(activation_function="relu")
        self.heads_parameters = [VHeadParameters(initializer="xavier")]
        self.rescale_gradient_from_head_by_factor = [1]
        self.optimizer_type = "Adam"
        self.batch_size = 256
        self.async_training = False
        self.learning_rate = 0.0003  # 3e-4 see appendix D in the paper
        # tau is set in SoftActorCriticAlgorithmParameters.rate_for_copying_weights_to_target
        self.create_target_network = True


# 2. Soft Q Value Network - SACCriticNetwork
# the whole network is built in the SACQHeadParameters. we use empty input embedder and middleware
class SACCriticNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {
            "observation": InputEmbedderParameters(scheme=EmbedderScheme.Empty)
        }
        self.middleware_parameters = FCMiddlewareParameters(scheme=MiddlewareScheme.Empty)
        self.heads_parameters = [
            SACQHeadParameters()
        ]  # SACQHeadParameters includes the topology of the head
        self.rescale_gradient_from_head_by_factor = [1]
        self.optimizer_type = "Adam"
        self.batch_size = 256
        self.async_training = False
        self.learning_rate = 0.0003
        self.create_target_network = False


# 3. policy Network
# Default parameters for policy network:
# topology :
#   input embedder : EmbedderScheme.Medium (Dense(256)) , relu activation
#   middleware : EmbedderScheme = [Dense(256)] , relu activation --> scheme should be overridden in preset
class SACPolicyNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {
            "observation": InputEmbedderParameters(activation_function="relu")
        }
        self.middleware_parameters = FCMiddlewareParameters(activation_function="relu")
        self.heads_parameters = [SACPolicyHeadParameters()]
        self.rescale_gradient_from_head_by_factor = [1]
        self.optimizer_type = "Adam"
        self.batch_size = 256
        self.async_training = False
        self.learning_rate = 0.0003
        self.create_target_network = False
        self.l2_regularization = 0  # weight decay regularization. not used in the original paper


# Algorithm Parameters


class SoftActorCriticAlgorithmParameters(AlgorithmParameters):
    """
    :param num_steps_between_copying_online_weights_to_target: (StepMethod)
        The number of steps between copying the online network weights to the target network weights.

    :param rate_for_copying_weights_to_target: (float)
        When copying the online network weights to the target network weights, a soft update will be used, which
        weight the new online network weights by rate_for_copying_weights_to_target. (Tau as defined in the paper)

    :param use_deterministic_for_evaluation: (bool)
        If True, during the evaluation phase, action are chosen deterministically according to the policy mean
        and not sampled from the policy distribution.
    """

    def __init__(self):
        super().__init__()
        self.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1)
        self.rate_for_copying_weights_to_target = 0.005
        # evaluate agent using deterministic policy (i.e. take the mean value)
        self.use_deterministic_for_evaluation = True


class SoftActorCriticAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(
            algorithm=SoftActorCriticAlgorithmParameters(),
            exploration=AdditiveNoiseParameters(),
            memory=ExperienceReplayParameters(),  # SAC doesnt use episodic related data
            # network wrappers:
            networks=OrderedDict(
                [
                    ("policy", SACPolicyNetworkParameters()),
                    ("q", SACCriticNetworkParameters()),
                    ("v", SACValueNetworkParameters()),
                ]
            ),
        )

    @property
    def path(self):
        return "markov.multi_agent_coach.agents.sac_agent:SoftActorCriticAgent"


# Soft Actor Critic - https://arxiv.org/abs/1801.01290
class SoftActorCriticAgent(PolicyOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union["LevelManager", "CompositeAgent"] = None):
        super().__init__(agent_parameters, parent)
        self.last_gradient_update_step_idx = 0

    def create_networks(self):
        """
        Create all the networks of the agent.
        The network creation will be done after setting the environment parameters for the agent, since they are needed
        for creating the network.

        :return: A list containing all the networks

        NOTE: we override this function (originally in agent.py) to build a modified computation graph where policy net
        and Q net are connected, policy_loss is added to policy net.

        """
        networks = {}
        _net_names = sorted(self.ap.network_wrappers.keys())
        if _net_names[0] == "policy" and _net_names[1] == "q" and _net_names[2] == "v":
            _net_names = ["policy", "q", "v"]
        for network_name in _net_names:
            if network_name == "q":
                self.ap.network_wrappers["q"].heads_parameters[0].P_action = (
                    networks["policy"].online_network.output_heads[0].actions
                )

            networks[network_name] = NetworkWrapper(
                name=network_name,
                agent_parameters=self.ap,
                has_target=self.ap.network_wrappers[network_name].create_target_network,
                has_global=self.has_global,
                spaces=self.spaces,
                replicated_device=self.replicated_device,
                worker_device=self.worker_device,
            )

            print(networks[network_name])
        if "policy" in _net_names:
            Q_state_keys = list(self.ap.network_wrappers["v"].input_embedders_parameters.keys())
            q_state_phs = dict()
            for q_state_key in Q_state_keys:
                q_state_phs[q_state_key] = networks["q"].online_network.inputs[q_state_key]

            # NOTE: check if Q_state_key is "stereo_camera"
            networks["policy"].online_network.output_heads[0]._build_loss(
                networks["q"].online_network.output_heads[0].q_output,
                networks["q"].online_network.output_heads[0].actions_ph,
                q_state_phs,
            )

            # build d_loss/d_weights
            policy_net_ = networks["policy"].online_network
            policy_net_.weighted_gradients.append(
                tf.gradients(policy_net_.output_heads[0].loss_tensor, policy_net_.weights)
            )
            # NOTE: check if P_action is connected to Q net and Q_action is connected to policy net

        return networks

    def learn_from_batch(self, batch):
        #####################
        # need to update the following networks:
        # 1. actor (policy)
        # 2. state value (v)
        # 3. critic (q1 and q2)
        # 4. target network - probably already handled by V

        #####################
        # define the networks to be used

        # State Value Network
        value_network = self.networks["v"]
        value_network_keys = self.ap.network_wrappers["v"].input_embedders_parameters.keys()

        # Critic Network
        q_network = self.networks["q"].online_network
        q_head = q_network.output_heads[0]
        q_network_keys = self.ap.network_wrappers["q"].input_embedders_parameters.keys()

        # Actor (policy) Network
        policy_network = self.networks["policy"].online_network
        policy_network_keys = self.ap.network_wrappers["policy"].input_embedders_parameters.keys()

        #######################################################
        # 1. updating the actor - according to (13) in the paper
        policy_inputs = copy.copy(batch.states(policy_network_keys))
        # DH: confirmed list(value_network_keys)[0]=="STEREO_CAMERAS"
        batch_size = len(policy_inputs[list(value_network_keys)[0]])
        num_actions = int(self.spaces.action.shape[0])
        # DH: feed pointer to Q online net's inputs['stereo_camera'] and Q online net's output_heads[0].actions_ph
        _action_mask = np.zeros((batch_size, num_actions + 1))
        _action_mask[:, num_actions] = 1

        initial_feed_dict = dict()
        for input_embedder_key in policy_network.output_heads[0].Q_state_ph.keys():
            initial_feed_dict[
                policy_network.output_heads[0].Q_state_ph[input_embedder_key]
            ] = policy_inputs[input_embedder_key]
        initial_feed_dict[policy_network.output_heads[0].Q_actions_ph] = _action_mask

        # DH: policy_network.weighted_gradients[6] is dPlicyLoss_dw.  see agents.py
        # DH: policy_network.output_heads[0].loss_tensor_unreduced is -value_target
        outputs_ = [
            policy_network.weighted_gradients[-1],
            policy_network.output_heads[0].loss_tensor_unreduced,
        ]
        # DH: temp change for debug
        outputs_ += policy_network.outputs

        policy_results = policy_network.predict(
            policy_inputs, outputs=outputs_, initial_feed_dict=initial_feed_dict
        )
        # policy_grads, neg_value_targets = policy_results
        (
            policy_grads,
            neg_value_targets,
            policy_mean_,
            policy_log_std_,
            raw_actions_,
            actions_,
            sampled_actions_logprob_,
            sampled_actions_logprob_mean_,
        ) = policy_results

        policy_network.apply_gradients(policy_grads)

        # ------------------------------------------------------------------------------------------------------
        ######################################################
        # 2. updating the state value online network weights
        # done by calculating the targets for the v head according to (5) in the paper
        # value_targets = log_targets-sampled_actions_logprob
        value_inputs = copy.copy(batch.states(value_network_keys))

        # NOTE: V net reuses this tensor to avoid redundant computation
        value_targets = -neg_value_targets

        # call value_network apply gradients with this target
        value_loss = value_network.online_network.train_on_batch(
            value_inputs, value_targets[:, None]
        )[0]

        ###################################################
        # 3. updating the critic (q networks)
        # updating q networks according to (7) in the paper

        # define the input to the q network: state has been already updated previously, but now we need
        # the actions from the batch (and not those sampled by the policy)
        q_inputs = copy.copy(batch.states(q_network_keys))
        batch_actions = batch.actions(len(batch.actions().shape) == 1)
        q_inputs["output_0_0"] = batch_actions

        # define the targets : scale_reward * reward + (1-terminal)*discount*v_target_next_state
        # define v_target_next_state
        value_inputs = copy.copy(batch.next_states(value_network_keys))
        v_target_next_state = value_network.target_network.predict(value_inputs)

        TD_targets = (
            batch.rewards(expand_dims=True)
            + (1.0 - batch.game_overs(expand_dims=True))
            * self.ap.algorithm.discount
            * v_target_next_state
        )

        # call critic network update
        result = q_network.train_on_batch(
            q_inputs, TD_targets, additional_fetches=[q_head.q1_loss, q_head.q2_loss]
        )
        total_loss, losses, unclipped_grads = result[:3]
        q1_loss, q2_loss = result[3]

        ####################################################
        # 4. updating the value target network
        # I just need to set the parameter rate_for_copying_weights_to_target in the agent parameters to be 1-tau
        # where tau is the hyper parameter as defined in sac original implementation
        return total_loss, losses, unclipped_grads

    def get_prediction(self, states):
        """
        get the mean and stdev of the policy distribution given 'states'
        :param states: the states for which we need to sample actions from the policy
        :return: mean and stdev
        """
        tf_input_state = self.prepare_batch_for_inference(states, "policy")
        return self.networks["policy"].online_network.predict(tf_input_state)

    def train(self):
        # since the algorithm works with experience replay buffer (non-episodic),
        # we cant use the policy optimization train method. we need Agent.train
        # note that since in Agent.train there is no apply_gradients, we need to do it in learn from batch
        return Agent.train(self)

    def choose_action(self, curr_state):
        """
        choose_action - chooses the most likely action
        if 'deterministic' - take the mean of the policy which is the prediction of the policy network.
        else - use the exploration policy
        :param curr_state:
        :return: action wrapped in ActionInfo
        """
        if not isinstance(self.spaces.action, BoxActionSpace):
            raise ValueError("SAC works only for continuous control problems")
        # convert to batch so we can run it through the network
        tf_input_state = self.prepare_batch_for_inference(curr_state, "policy")
        # use the online network for prediction
        policy_network = self.networks["policy"].online_network
        policy_head = policy_network.output_heads[0]
        result = policy_network.predict(
            tf_input_state, outputs=[policy_head.policy_mean, policy_head.actions]
        )
        action_mean, action_sample = result

        # if using deterministic policy, take the mean values. else, use exploration policy to sample from the pdf
        if self.phase == RunPhase.TEST and self.ap.algorithm.use_deterministic_for_evaluation:
            action = action_mean[0]
        else:
            action = action_sample[0]

        action_info = ActionInfo(action=action)
        return action_info
