import logging
import os
import sys

import numpy as np
import tensorflow as tf

from . import ModeKeys, Module, layers
from .core import Fake
from .core.model import Model
from .layers.ops import get_tf_vars_list


class ResNetXXModel(Model):

    LAYER_COUNTS = {18: 40}

    _PARAMS = {
        "batch_size": 128,
        "loss_scale": 1,
        "dtype": "float32",
        "fine_tune": False,
        "remove_layers": None,
        "momentum": 0.9,
        "data_format": "channels_first",
        "temperature": 3,
        "distillation_coefficient": 1,
        "weight_decay": 2e-4,
    }

    @staticmethod
    def learning_rate_with_decay(
        batch_size, batch_denom, num_images, boundary_epochs, decay_rates, base_lr=0.1, warmup=False
    ):
        """Get a learning rate that decays step-wise as training progresses.

        Args:
            batch_size: the number of examples processed in each training batch.
            batch_denom: this value will be used to scale the base learning rate.
            `0.1 * batch size` is divided by this number, such that when
            batch_denom == batch_size, the initial learning rate will be 0.1.
            num_images: total number of images that will be used for training.
            boundary_epochs: list of ints representing the epochs at which we
            decay the learning rate.
            decay_rates: list of floats representing the decay rates to be used
            for scaling the learning rate. It should have one more element
            than `boundary_epochs`, and all elements should have the same type.
            base_lr: Initial learning rate scaled based on batch_denom.
            warmup: Run a 5 epoch warmup to the initial lr.
        Returns:
            Returns a function that takes a single argument - the number of batches
            trained so far (global_step)- and returns the learning rate to be used
            for training the next batch.
        """
        initial_learning_rate = base_lr * batch_size / batch_denom
        batches_per_epoch = num_images / batch_size
        # Reduce the learning rate at certain epochs.
        # CIFAR-10: divide by 10 at epoch 100, 150, and 200
        boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
        vals = [initial_learning_rate * decay for decay in decay_rates]

        def learning_rate_fn(global_step):
            """Builds scaled learning rate function with 5 epoch warm up."""
            lr = tf.train.piecewise_constant(global_step, boundaries, vals)
            if warmup:
                warmup_steps = int(batches_per_epoch * 5)
                warmup_lr = (
                    initial_learning_rate
                    * tf.cast(global_step, tf.float32)
                    / tf.cast(warmup_steps, tf.float32)
                )
                return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
            return lr

        return learning_rate_fn

    @staticmethod
    def builder(features, labels, mode, params):
        """Shared functionality for different resnet model_fns.

        Initializes the ResnetModel representing the model layers
        and uses that model to build the necessary EstimatorSpecs for
        the `mode` in question. For training, this means building losses,
        the optimizer, and the train op that get passed into the EstimatorSpec.
        For evaluation and prediction, the EstimatorSpec is returned without
        a train op, but with the necessary parameters for the given mode.

        Args:
            features: tensor representing input images
            labels: tensor representing class labels for all input images
            mode: current estimator mode; should be one of
            `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`

            Everything else goes in the params parameter:

            model_class: a class representing a TensorFlow model that has a __call__
            function. We assume here that this is a subclass of ResnetModel.
            learning_rate_fn: function that returns the current learning rate given
            the current global_step
            data_format: Input format ('channels_last', 'channels_first', or None).
            If set to None, the format is dependent on whether a GPU is available.
            fine_tune: If True only train the dense layers(final layers).
            remove_layers: A Boolean array of layers to remove.

        Returns:
            EstimatorSpec parameterized according to the input params and the
            current mode.
        """
        model_class = params["model_class"]
        data_format = params["data_format"]
        num_classes = params["num_classes"]
        loss_scale = params["loss_scale"]
        resnet_size = params["resnet_size"]
        fine_tune = params["fine_tune"]
        remove_layers = params["remove_layers"]
        dtype = params["dtype"]
        name = params["name"]
        loss_filter_fn = None
        num_images = params["num_images"]
        batch_size = params["batch_size"]
        momentum = params["momentum"]
        teacher = params["teacher"]
        weights = params["weights"]
        params_scope = params["params_scope"]
        temperature = params["temperature"]
        distillation_coefficient = params["distillation_coefficient"]
        weight_decay = params["weight_decay"]

        if mode == ModeKeys.REFERENCE:
            fake = True
        else:
            fake = False
        if not fake:
            # Generate a summary node for the images
            tf.summary.image("images", features, max_outputs=6)
            # Checks that features/images have same data type being used for calculations.
            assert features.dtype == dtype

        model = model_class(
            name=name,
            resnet_size=18,
            bottleneck=False,
            num_classes=num_classes,
            num_filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            block_strides=[(resnet_size - 2) // 6] * 3,
            block_sizes=[1, 2, 2],
            data_format=data_format,
        )

        training = True if mode == ModeKeys.TRAIN else False

        logits = model(
            inputs=features,
            training=training,
            remove_layers=remove_layers,
            weights=weights,
            fake=fake,
            params_scope=params_scope,
        )

        if mode == ModeKeys.REFERENCE:
            return model

        assert not fake
        if not teacher is None and not mode == ModeKeys.PREDICT:
            # Build a teacher model using the same loaded weights.
            teacher_model = model_class(
                name=teacher,
                resnet_size=18,
                bottleneck=False,
                num_classes=num_classes,
                num_filters=16,
                kernel_size=3,
                conv_stride=1,
                first_pool_size=None,
                first_pool_stride=None,
                block_strides=[(resnet_size - 2) // 6] * 3,
                block_sizes=[1, 2, 2],
                data_format=data_format,
            )

            teacher_logits = teacher_model(
                inputs=features,
                training=False,
                remove_layers=None,
                weights=weights,
                params_scope=params_scope,
            )

        # This acts as a no-op if the logits are already in fp32 (provided logits are
        # not a SparseTensor). If dtype is is low precision, logits must be cast to
        # fp32 for numerical stability.
        logits = tf.cast(logits, tf.float32)
        if not teacher is None and not mode == ModeKeys.PREDICT:
            teacher_logits = tf.cast(teacher_logits, tf.float32)

        predictions = {
            "classes": tf.argmax(logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        }

        if not teacher is None and not mode == ModeKeys.PREDICT:
            predictions["teacher_classes"] = tf.argmax(teacher_logits, axis=1)
            predictions["teacher_probabilities"] = tf.nn.softmax(
                teacher_logits, name="teacher_softmax_tensor"
            )

        if mode == ModeKeys.PREDICT:
            # Return the predictions and the specification for serving a SavedModel
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={"predict": tf.estimator.export.PredictOutput(predictions)},
            )

        # Calculate loss, which includes softmax cross entropy and L2 regularization.
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

        if teacher is not None:
            distillation_loss = ResNetXXModel.create_distillation_loss(
                logits, teacher_logits, temperature
            )
            teacher_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                logits=teacher_logits, labels=labels
            )

        # Create a tensor named cross_entropy and distillation loss for logging purposes.
        tf.identity(cross_entropy, name="cross_entropy")
        tf.summary.scalar("cross_entropy", cross_entropy)

        if teacher is not None:
            tf.identity(distillation_loss, name="distillation_loss")
            tf.summary.scalar("distillation_loss", distillation_coefficient * distillation_loss)

            tf.identity(cross_entropy, name="teacher_cross_entropy")
            tf.summary.scalar("teacher_cross_entropy", teacher_cross_entropy)

        # If no loss_filter_fn is passed, assume we want the default behavior,
        # which is that batch_normalization variables are excluded from loss.
        def exclude_batch_norm(name):
            return "batch_normalization" not in name

        loss_filter_fn = loss_filter_fn or exclude_batch_norm

        learning_rate_fn = ResNetXXModel.learning_rate_with_decay(
            base_lr=0.1,
            batch_size=batch_size,
            batch_denom=128,
            num_images=num_images["train"],
            boundary_epochs=[20, 30],
            decay_rates=[0.1, 0.01, 0.001],
        )

        def loss_filter_fn(_):
            return True

        # Add weight decay to the loss.
        trainable_vars = get_tf_vars_list(name)

        l2_loss = weight_decay * tf.add_n(
            # loss is computed using fp32 for numerical stability.
            [
                tf.nn.l2_loss(tf.cast(v, tf.float32))
                for v in trainable_vars
                if loss_filter_fn(v.name)
            ]
        )
        tf.summary.scalar("l2_loss", l2_loss)
        loss = cross_entropy + l2_loss
        if teacher is not None:
            loss = loss + distillation_coefficient * distillation_loss

        if mode == ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()

            learning_rate = learning_rate_fn(global_step)

            # Create a tensor named learning_rate for logging purposes
            tf.identity(learning_rate, name="learning_rate")
            tf.summary.scalar("learning_rate", learning_rate)

            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)

            def _dense_grad_filter(gvs):
                """Only apply gradient updates to the final layer.

                This function is used for fine tuning.

                Args:
                    gvs: list of tuples with gradients and variable info
                Returns:
                    filtered gradients so that only the dense layer remains
                """
                return [(g, v) for g, v in gvs if "dense" in v.name]

            if loss_scale != 1:
                # When computing fp16 gradients, often intermediate tensor values are
                # so small, they underflow to 0. To avoid this, we multiply the loss by
                # loss_scale to make these tensor values loss_scale times bigger.
                scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)
            assert not fine_tune
            if fine_tune:
                scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

                # Once the gradient computation is complete we can scale the gradients
                # back to the correct scale before passing them to the optimizer.
                unscaled_grad_vars = [(grad / loss_scale, var) for grad, var in scaled_grad_vars]
                minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
            else:
                grad_vars = optimizer.compute_gradients(loss, var_list=trainable_vars)
            if fine_tune:
                grad_vars = _dense_grad_filter(grad_vars)
            minimize_op = optimizer.apply_gradients(grad_vars, global_step)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group(minimize_op, update_ops)
        else:
            train_op = None

        accuracy = tf.metrics.accuracy(labels, predictions["classes"])
        accuracy_top_5 = tf.metrics.mean(
            tf.nn.in_top_k(predictions=logits, targets=labels, k=5, name="top_5_op")
        )

        if not teacher is None:
            teacher_accuracy = tf.metrics.accuracy(labels, predictions["teacher_classes"])
            teacher_accuracy_top_5 = tf.metrics.mean(
                tf.nn.in_top_k(
                    predictions=teacher_logits, targets=labels, k=5, name="teacher_top_5_op"
                )
            )
        metrics = {"accuracy": accuracy, "accuracy_top_5": accuracy_top_5}

        if not teacher is None:
            metrics["teacher_accuracy"] = teacher_accuracy
            metrics["teacher_accuracy_top_5"] = teacher_accuracy_top_5

        # Create a tensor named train_accuracy for logging purposes
        tf.identity(accuracy[1], name="train_accuracy")
        tf.identity(accuracy_top_5[1], name="train_accuracy_top_5")
        tf.summary.scalar("train_accuracy", accuracy[1])
        tf.summary.scalar("train_accuracy_top_5", accuracy_top_5[1])

        if not teacher is None:
            tf.summary.scalar("teacher_accuracy", teacher_accuracy[1])
            tf.summary.scalar("teacher_accuracy_top_5", teacher_accuracy_top_5[1])

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics,
        )

    @staticmethod
    def create_distillation_loss(logit_1, logit_2, temperature):
        """Creates a distilaltion loss"""
        knowledge_1 = ResNetXXModel.temperature_softmax(logit_1, temperature)
        knowledge_2 = ResNetXXModel.temperature_softmax(logit_2, temperature)
        return ResNetXXModel.L2(knowledge_1, knowledge_2)

    @staticmethod
    def temperature_softmax(input, temperature=1):
        """
        Creates the softmax normalization
        Args:
            input: Where is the input of the layer coming from
            temperature: Temperature of the softmax.
        Returns:
            softmax values.
        """
        return tf.nn.softmax(tf.div(input, temperature))

    @staticmethod
    def L2(node1, node2):
        """Returns a symbol thats the 2-norm error"""
        return tf.reduce_mean(tf.squared_difference(node1, node2))

    @staticmethod
    def block_layer(
        inputs,
        filters,
        bottleneck,
        block_module,
        blocks,
        strides,
        weights,
        training,
        name,
        data_format,
        layer_count,
        remove_layers,
        weight_scope,
        fake=False,
    ):
        """Creates one layer of blocks for the ResNet model.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
                [batch, height_in, width_in, channels] depending on data_format.
            filters: The number of filters for the first convolution of the layer.
            bottleneck: Is the block created a bottleneck block.
            block_module: The block to use within the model, either `building_block` or
                `bottleneck_block`. Only BuildingBlock is allowed now.
            blocks: The number of blocks contained in the layer.
            strides: The stride to use for the first convolution of the layer. If
                greater than 1, this layer will ultimately downsample the input.
            training: Either True or False, whether we are currently training the
                model. Needed for batch norm.
            weights: Parameters in numpy format.
            name: A string name for the tensor output of the block layer.
            data_format: The input format ('channels_last' or 'channels_first').
            layer_count: Supply the current `layer_count` from the on-going construction.
            remove_layers: Supply the current `remove_layers` array as is.

        Returns:
            The output tensor of the block layer.
        """
        resnet_block_module = Module(name)
        block_count = 0

        filters_out = filters * 4 if bottleneck else filters

        def projection_shortcut(
            inputs, layer_count_ps, strides_ps, weights, weight_scope, fake=False
        ):
            return (
                layers.Conv(
                    name=str(layer_count_ps),
                    inputs=inputs,
                    filters=filters_out,
                    kernel_size=1,
                    strides=strides_ps,
                    weights=weights,
                    data_format=data_format,
                    weight_scope=weight_scope,
                    start=layer_count_ps,
                    fake=fake,
                    end=layer_count_ps,
                ),
                layer_count_ps + 1,
            )  # This end should change.

        block = block_module(
            name + "_mini_block_" + str(block_count),
            inputs,
            filters,
            training,
            projection_shortcut,
            strides,
            data_format,
            layer_count,
            remove_layers,
            weights,
            weight_scope,
            fake,
        )
        resnet_block_module.add_module(block, block.get_name())
        inputs = block.output
        block_count += 1
        layer_count = block.get_layer_count()

        for _ in range(1, blocks):
            block = block_module(
                name + "_mini_block_" + str(block_count),
                inputs,
                filters,
                training,
                None,
                1,
                data_format,
                layer_count,
                remove_layers,
                weights,
                weight_scope,
                fake,
            )
            resnet_block_module.add_module(block, block.get_name())
            block_count += 1
            layer_count = block.get_layer_count()
        return (resnet_block_module, layer_count)


class ResNetBase(object):
    def __init__(
        self,
        name,
        resnet_size,
        bottleneck,
        num_classes,
        num_filters,
        kernel_size,
        conv_stride,
        first_pool_size,
        first_pool_stride,
        block_sizes,
        block_strides,
        data_format="channels_first",
    ):
        """
        This a demo for a temporaray Network object.
        Args:
            name: Some name for the network definition.
            images: A placeholder in the tf graph for images.
            labels: A placeholder in the tf graph for lablels.
            resnet_size: A single integer for the size of the ResNet model.
            bottleneck: Use regular blocks or bottleneck blocks.
            num_classes: The number of classes used as labels.
            num_filters: The number of filters to use for the first block layer
                of the model. This number is then doubled for each subsequent block
                layer.
            kernel_size: The kernel size to use for convolution.
            conv_stride: stride size for the initial convolutional layer
            first_pool_size: Pool size to be used for the first pooling layer.
                If none, the first pooling layer is skipped.
            first_pool_stride: stride size for the first pooling layer. Not used
                if first_pool_size is None.
            block_sizes: A list containing n values, where n is the number of sets of
                block layers desired. Each value should be the number of blocks in the
                i-th set.
            block_strides: List of integers representing the desired stride size for
                each of the sets of block layers. Should be same length as block_sizes.
            final_size: The expected size of the model after the second pooling.
            version: Integer representing which version of the ResNet network to use.
                See README for details. Valid values: [1, 2]
            data_format: Input format ('channels_last', 'channels_first', or None).
                If set to None, the format is dependent on whether a GPU is available.
            dtype: The TensorFlow dtype to use for calculations. If not specified
                tf.float32 is used.
        """
        if resnet_size is not 18:
            raise NotImplementedError("Only ResNet 18s are available at the moment.")

        self.resnet_size = resnet_size
        if bottleneck:
            raise NotImplementedError("Bottleneck layers not available at the moment.")
        else:
            self.block_module = layers.BuildingBlock

        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.bottleneck = bottleneck
        self.dtype = tf.float32
        self.name = name

    def pretty_print(self):
        """Prints important characters of the class"""
        logging.info("Num Classes: " + str(self.num_classes))
        logging.info("Num Filters: " + str(self.num_filters))
        logging.info("Kernel Size: " + str(self.kernel_size))
        logging.info("Conv Stride: " + str(self.conv_stride))
        logging.info(
            "Pool 1 size: " + str(self.first_pool_size) + " Stride: " + str(self.first_pool_stride)
        )
        logging.info("Block sizes: " + str(self.block_sizes))
        logging.info("Bottleneck: " + str(self.bottleneck))
        logging.info("Resnet Size: " + str(self.resnet_size))

    def _model_variable_scope(self):
        """Returns the scope of the model"""
        return tf.variable_scope(self.name)

    def initialize(params):
        """Will reinitialize all layers with the provided parameters.

        Args:
            params: Initialize with these parameters
        """
        raise NotImplementedError

    def __call__(
        self, inputs, training, remove_layers=None, weights=None, params_scope=None, fake=False
    ):
        """Add operations to classify a batch of input images.

        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Set to True to add operations required only when
                training the classifier.
            remove_layers: A vector of booleans, one-per-layer.
            weights: A pickle file containing a dictionary of layer names and numpy weights.
            params_scope: A scope for the dictionary of weights in the pickle file.
            fake: If True, the graph won't be tensorflow and won't be trainable. Use this for
                reference models and such.
        Returns:
            A logits Tensor with shape [<batch_size>, self.num_classes].
        """
        self.net = Module(self.name)

        if remove_layers is None:
            remove_layers = [False] * ResNetXXModel.LAYER_COUNTS[self.resnet_size]
        else:
            if not len(remove_layers) == ResNetXXModel.LAYER_COUNTS[self.resnet_size]:
                raise ValueError(
                    "remove_layers must have "
                    + str(ResNetXXModel.LAYER_COUNTS[self.resnet_size])
                    + " elements."
                )

        logging.info("Creating network: " + self.name)

        layer_count = 0
        with self._model_variable_scope():
            if self.data_format == "channels_first":
                if not fake:
                    inputs = tf.transpose(inputs, [0, 3, 1, 2])
                else:
                    inputs = Fake(
                        (inputs.shape[0], inputs.shape[3], inputs.shape[1], inputs.shape[2])
                    )

            if not remove_layers[layer_count]:
                l = layers.Conv(
                    name=str(layer_count),
                    inputs=inputs,
                    filters=self.num_filters,
                    kernel_size=self.kernel_size,
                    strides=self.conv_stride,
                    data_format=self.data_format,
                    start=layer_count,
                    weights=weights,
                    fake=fake,
                    weight_scope=params_scope,
                    end=layer_count,
                )
                inputs = l.output
                if not fake:
                    inputs = tf.identity(inputs, "initial_conv")
                self.net.add_module(l, l.get_name())
            layer_count += 1

            if self.first_pool_size and not remove_layers[layer_count]:
                l = layers.Pool(
                    name=str(layer_count),
                    inputs=inputs,
                    pool_size=self.first_pool_size,
                    strides=self.first_pool_stride,
                    padding="SAME",
                    fake=fake,
                    data_format=self.data_format,
                )
                inputs = l.output
                if not fake:
                    inputs = tf.identity(inputs, "initial_max_pool")
                self.net.add_module(l, l.get_name())
            layer_count += 1

            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2 ** i)
                block, layer_count = ResNetXXModel.block_layer(
                    inputs=inputs,
                    filters=num_filters,
                    bottleneck=self.bottleneck,
                    block_module=self.block_module,
                    blocks=num_blocks,
                    strides=self.block_strides[i],
                    training=training,
                    weights=weights,
                    name="block_layer_{}".format(i + 1),
                    data_format=self.data_format,
                    layer_count=layer_count,
                    remove_layers=remove_layers,
                    weight_scope=params_scope,
                    fake=fake,
                )
                inputs = block.output
                self.net.add_module(block, block.get_name())

            if not remove_layers[layer_count]:
                l = layers.BatchNorm(
                    name=str(layer_count),
                    inputs=inputs,
                    training=training,
                    weights=weights,
                    data_format=self.data_format,
                    start=layer_count,
                    weight_scope=params_scope,
                    fake=fake,
                    end=layer_count,
                )
                inputs = l.output
                self.net.add_module(l, l.get_name())
            layer_count += 1

            if not remove_layers[layer_count]:
                l = layers.ReLU(
                    name=str(layer_count),
                    inputs=inputs,
                    fake=fake,
                    start=layer_count,
                    end=layer_count,
                )
                inputs = l.output
                self.net.add_module(l, l.get_name())
            layer_count += 1

            if not fake:
                axes = [2, 3] if self.data_format == "channels_first" else [1, 2]
                inputs = tf.reduce_mean(inputs, axes, keepdims=True)
                inputs = tf.identity(inputs, "final_reduce_mean")

                inputs = tf.squeeze(inputs, axes)
            else:
                inputs = Fake((None, inputs.shape[1]))
            l = layers.Dense(
                name=str(layer_count),
                inputs=inputs,
                units=self.num_classes,
                start=layer_count,
                weights=weights,
                fake=fake,
                weight_scope=params_scope,
                end=layer_count,
            )
            self.net.add_module(l, l.get_name())
            inputs = l.output
            if not fake:
                inputs = tf.identity(inputs, "logits")

            return inputs


class ResNet18Model(ResNetXXModel):
    def __init__(self, name, model_params, scope=""):
        self.params = ResNetXXModel._PARAMS
        self.params["model_class"] = self.definition
        self.params["resnet_size"] = 18
        self.size = 18
        for k in model_params.keys():
            self.params[k] = model_params[k]

    definition = ResNetBase
