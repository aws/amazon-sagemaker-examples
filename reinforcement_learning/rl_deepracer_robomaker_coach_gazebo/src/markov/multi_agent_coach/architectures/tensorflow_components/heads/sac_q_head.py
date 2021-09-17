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

import tensorflow as tf
from rl_coach.architectures.tensorflow_components.heads.head import Head
from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import QActionStateValue
from rl_coach.spaces import BoxActionSpace, SpacesDefinition


class SACQHead(Head):
    def __init__(
        self,
        agent_parameters: AgentParameters,
        spaces: SpacesDefinition,
        network_name: str,
        head_idx: int = 0,
        loss_weight: float = 1.0,
        is_local: bool = True,
        activation_function: str = "relu",
        dense_layer=Dense,
        output_bias_initializer=None,
        P_action=None,
    ):
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
        self.name = "q_values_head"
        if isinstance(self.spaces.action, BoxActionSpace):
            self.num_actions = self.spaces.action.shape  # continuous actions
        else:
            raise ValueError(
                "SACQHead does not support action spaces of type: {class_name}".format(
                    class_name=self.spaces.action.__class__.__name__,
                )
            )
        self.return_type = QActionStateValue
        # extract the topology from the SACQHeadParameters
        # DH: [256,256]
        self.network_layers_sizes = (
            agent_parameters.network_wrappers["q"].heads_parameters[0].network_layers_sizes
        )
        self.output_bias_initializer = output_bias_initializer
        self.P_action = P_action

    def _build_module(self, input_layer):
        # SAC Q network is basically 2 networks running in parallel on the same input (state , action)
        # state is the observation fed through the input_layer, action is fed through placeholder to the header
        # each is calculating q value  : q1(s,a) and q2(s,a)
        # the output of the head is min(q1,q2)
        self.actions_ph = tf.placeholder(tf.float32, [None, self.num_actions + 1], name="action")
        self.actions = (
            tf.slice(self.actions_ph, [0, 0], [-1, int(self.num_actions)])
            + tf.slice(self.actions_ph, [0, int(self.num_actions)], [-1, 1]) * self.P_action
        )
        self.target = tf.placeholder(tf.float32, [None, 1], name="q_targets")
        self.input = [self.actions]
        self.output = []
        # Note (1) : in the author's implementation of sac (in rllab) they summarize the embedding of observation and
        # action (broadcasting the bias) in the first layer of the network.

        # build q1 network head
        with tf.variable_scope("q1_head"):

            qi_obs_act_emb = tf.concat([input_layer, self.actions], -1)

            layer_size = self.network_layers_sizes[0]

            qi_output = self.dense_layer(layer_size)(
                qi_obs_act_emb, activation=self.activation_function
            )

            for layer_size in self.network_layers_sizes[1:]:
                qi_output = self.dense_layer(layer_size)(
                    qi_output, activation=self.activation_function
                )
            # the output layer
            self.q1_output = self.dense_layer(1)(
                qi_output, name="q1_output", bias_initializer=self.output_bias_initializer
            )

        # build q2 network head
        with tf.variable_scope("q2_head"):

            self.input_layer_raw = input_layer

            qi_obs_act_emb = tf.concat([input_layer, self.actions], -1)
            self.qi_obs_act_emb = qi_obs_act_emb
            layer_size = self.network_layers_sizes[0]
            qi_output = self.dense_layer(layer_size)(
                qi_obs_act_emb, activation=self.activation_function
            )

            # qi_obs_emb = self.dense_layer(layer_size)(input_layer, activation=self.activation_function)
            # self.qi_obs_emb = qi_obs_emb
            # qi_act_emb = self.dense_layer(layer_size)(self.actions, activation=self.activation_function)
            # qi_output = qi_obs_emb + qi_act_emb     # merging the inputs by summarizing them (see Note (1))
            for layer_size in self.network_layers_sizes[1:]:
                qi_output = self.dense_layer(layer_size)(
                    qi_output, activation=self.activation_function
                )
            # the output layer
            self.q2_output = self.dense_layer(1)(
                qi_output, name="q2_output", bias_initializer=self.output_bias_initializer
            )

        # take the minimum as the network's output. this is the log_target (in the original implementation)
        self.q_output = tf.minimum(self.q1_output, self.q2_output, name="q_output")
        # the policy gradients
        # self.q_output_mean = tf.reduce_mean(self.q1_output)         # option 1: use q1
        self.q_output_mean = tf.reduce_mean(self.q_output)  # option 2: use min(q1,q2)

        self.output.append(self.q_output)
        self.output.append(self.q_output_mean)
        # DH: debug only
        self.output.append(self.actions)
        self.output.append(self.input_layer_raw)
        self.output.append(self.qi_obs_act_emb)

        # defining the loss
        self.q1_loss = 0.5 * tf.reduce_mean(tf.square(self.q1_output - self.target))
        self.q2_loss = 0.5 * tf.reduce_mean(tf.square(self.q2_output - self.target))
        # eventually both losses are depends on different parameters so we can sum them up
        self.loss = self.q1_loss + self.q2_loss
        tf.losses.add_loss(self.loss)

    def __str__(self):
        result = [
            "q1 output" "\t\tDense (num outputs = 256)",
            "\t\tDense (num outputs = 256)",
            "\t\tDense (num outputs = 1)",
            "q2 output" "\t\tDense (num outputs = 256)",
            "\t\tDense (num outputs = 256)",
            "\t\tDense (num outputs = 1)",
            "min(Q1,Q2)",
        ]
        return "\n".join(result)
