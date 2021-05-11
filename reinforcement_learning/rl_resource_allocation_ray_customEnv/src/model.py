import tensorflow as tf

try:
    import tensorflow.contrib.slim as slim
except ImportError:
    import tf_slim as slim
try:
    from ray.rllib.models.misc import normc_initializer
except ImportError:
    from ray.rllib.models.tf.misc import normc_initializer

from ray.rllib.models import Model, ModelCatalog


class ActionMaskModel(Model):
    """
    Model that only allows valid actions
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        mask = input_dict["obs"]["action_mask"]

        last_layer = input_dict["obs"]["real_obs"]
        hiddens = options["fcnet_hiddens"]
        for i, size in enumerate(hiddens):
            label = "fc{}".format(i)
            last_layer = slim.fully_connected(
                last_layer,
                size,
                weights_initializer=normc_initializer(1.0),
                activation_fn=tf.nn.tanh,
                scope=label,
            )
        action_logits = slim.fully_connected(
            last_layer,
            num_outputs,
            weights_initializer=normc_initializer(0.01),
            activation_fn=None,
            scope="fc_out",
        )

        if num_outputs == 1:
            return action_logits, last_layer

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(mask), tf.float32.min)
        masked_logits = inf_mask + action_logits

        return masked_logits, last_layer


def register_actor_mask_model():
    ModelCatalog.register_custom_model("action_mask", ActionMaskModel)
