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

import numpy as np
import tensorflow as tf
from rl_coach.architectures.tensorflow_components.heads.head import Head
from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import ActionProbabilities
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import eps


class SACPolicyHead(Head):
    def __init__(
        self,
        agent_parameters: AgentParameters,
        spaces: SpacesDefinition,
        network_name: str,
        head_idx: int = 0,
        loss_weight: float = 1.0,
        is_local: bool = True,
        activation_function: str = "relu",
        squash: bool = True,
        dense_layer=Dense,
        sac_alpha=0.2,
        rescale_action_values=True,
        log_std_bounds=[-20, 2],
    ):  # q_network
        super().__init__(
            agent_parameters,
            spaces,
            network_name,
            head_idx,
            loss_weight,
            is_local,
            activation_function,
            dense_layer=dense_layer,
        )
        self.name = "sac_policy_head"
        self.return_type = ActionProbabilities
        self.num_actions = self.spaces.action.shape  # continuous actions
        self.squash = squash  # squashing using tanh
        self.sac_alpha = sac_alpha
        self.rescale_action_values = rescale_action_values
        self.log_std_bounds = log_std_bounds

    def _build_module(self, input_layer):
        # DH TODO: check shape of self.num_actions. It should be int
        self.given_raw_actions = tf.placeholder(
            tf.float32, [None, self.num_actions], name="actions"
        )
        self.input = [self.given_raw_actions]
        self.output = []

        # build the network
        self._build_continuous_net(input_layer, self.spaces.action)

    def _build_continuous_net(self, input_layer, action_space):
        num_actions = action_space.shape[0]
        # ----------DH:Dense(256) in middle ware + Dense(256) + separate dense layers for mu and logsig----
        mu_and_logsig = self.dense_layer(256)(input_layer, activation="relu")
        self.policy_mean = tf.identity(
            self.dense_layer(num_actions)(mu_and_logsig, name="policy_mean"), name="policy"
        )
        self.policy_log_std = tf.clip_by_value(
            self.dense_layer(num_actions)(mu_and_logsig),
            self.log_std_bounds[0],
            self.log_std_bounds[1],
            name="policy_log_std",
        )
        # ------------------------------------------------------------------------------------------------

        # define the distributions for the policy
        # Tensorflow's multivariate normal distribution supports reparameterization
        tfd = tf.contrib.distributions
        self.policy_distribution = tfd.MultivariateNormalDiag(
            loc=self.policy_mean, scale_diag=tf.exp(self.policy_log_std)
        )

        # define network outputs
        # note that tensorflow supports reparametrization.
        # i.e. policy_action_sample is a tensor through which gradients can flow
        self.raw_actions = self.policy_distribution.sample()

        if self.squash:
            self.actions = tf.tanh(self.raw_actions)
        else:
            self.actions = self.raw_actions

        # policy_action_logprob is a tensor through which gradients can flow
        # -------- DH the log_prob should be computed considering the action scale --------------------------------
        self.action_scale = (self.spaces.action.high - self.spaces.action.low) / 2.0
        self.action_bias = (self.spaces.action.high + self.spaces.action.low) / 2.0
        if self.rescale_action_values:
            # adjust raw_mean to mean by tanh and scale; apply scale to actions; squash trick on prob. with scale
            # NOTE: self.action used to compute sampled_actions_logprob is tanh(self.raw_action)
            self.sampled_actions_logprob = self.policy_distribution.log_prob(
                self.raw_actions
            ) - tf.reduce_sum(tf.log(self.action_scale * (1 - self.actions ** 2) + eps), axis=1)
            self.policy_mean = self.action_scale * tf.tanh(self.policy_mean) + self.action_bias
            self.actions = self.action_scale * self.actions + self.action_bias
        else:
            self.sampled_actions_logprob = self.policy_distribution.log_prob(
                self.raw_actions
            ) - tf.reduce_sum(tf.log((1 - self.actions ** 2) + eps), axis=1)
            self.policy_mean = tf.tanh(self.policy_mean)
        # ---------------------------------------------------------------------------------------------------------
        self.sampled_actions_logprob_mean = tf.reduce_mean(self.sampled_actions_logprob)
        # ----- DH: output logvar and ajusted mean-------------
        self.output.append(self.policy_mean)  # output[0]
        self.output.append(self.policy_log_std)  # output[1]
        # ------------------------------------------------------

        self.output.append(self.raw_actions)  # output[2] : sampled raw action (before squash)
        self.output.append(
            self.actions
        )  # output[3] : squashed (if needed) version of sampled raw_actions
        self.output.append(
            self.sampled_actions_logprob
        )  # output[4]: log prob of sampled action (squash corrected)
        # output[5]: mean of log prob of sampled actions (squash corrected)
        self.output.append(self.sampled_actions_logprob_mean)

    def _build_loss(self, Q_output, Q_actions_ph, Q_state_ph):
        # DH: NOTE: self.loss_tensor is actual loss, i.e. a scalar, \ while
        #  self.loss_tensor_unreduced is of shape (batch_size,), which is -value_target for V net
        self.loss_tensor_unreduced = self.sac_alpha * self.sampled_actions_logprob - tf.squeeze(
            Q_output
        )
        self.loss_tensor = tf.reduce_mean(self.loss_tensor_unreduced)
        # DH: add the pointer to Q's placeholder: Replay_Action
        self.Q_actions_ph = Q_actions_ph
        self.Q_state_ph = Q_state_ph

    def __str__(self):
        result = [
            "policy head:" "\t\tDense (num outputs = 256)",
            "\t\tDense (num outputs = 256)",
            "\t\tDense (num outputs = {0})".format(2 * self.num_actions),
            "policy_mu = output[:num_actions], policy_std = output[num_actions:]",
        ]
        return "\n".join(result)
