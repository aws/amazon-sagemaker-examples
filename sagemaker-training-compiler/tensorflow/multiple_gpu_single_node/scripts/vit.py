import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import argparse, os
from PIL import Image
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import warnings


mixed_precision.set_global_policy("mixed_float16")

# Setting seed for reproducibiltiy
SEED = 42
keras.utils.set_random_seed(SEED)

strategy = tf.distribute.MirroredStrategy()


def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype(np.float64)
    for i in range(3):
        minval = arr[..., i].min()
        maxval = arr[..., i].max()
        if minval != maxval:
            arr[..., i] -= minval
            arr[..., i] *= 255.0 / (maxval - minval)
    return arr.astype(np.uint8)


def resize(INPUT_SHAPE, img):
    """Resize image to specified size.

    Resize shorter side to specified shape while maintaining aspect ratio.
    """
    aspect_ratio = img.size[0] / img.size[1]
    _size = [0, 0]
    if img.size[0] < img.size[1]:
        _size[0] = INPUT_SHAPE[0]
        _size[1] = int(np.ceil(_size[0] / aspect_ratio))
    else:
        _size[1] = INPUT_SHAPE[1]
        _size[0] = int(np.ceil(_size[1] * aspect_ratio))
    return img.resize(tuple(_size))


def load_dataset(INPUT_SHAPE, NUM_CLASSES):
    """Load the Caltech-256 dataset from SageMaker input directory.

    The images are expected to be .jpg format stored under directories
    that indicate their object category. Images smaller than the specified
    size are ignored.

    Qualifying images are then resized and center cropped to meet the
    size criterion specificed. Labels are obtained from the directory structure.
    """
    x_train, y_train = [], []
    for root, dirs, files in os.walk(os.environ["SM_INPUT_DIR"]):
        for file in [f for f in files if f.endswith(".jpg")]:
            fpath = os.path.join(root, file)
            with Image.open(fpath) as img:
                if img.size[0] < INPUT_SHAPE[0] or img.size[1] < INPUT_SHAPE[1]:
                    continue
                else:
                    img = resize(INPUT_SHAPE, img)
                    array = np.asarray(img)
                    margin = [0, 0]
                    for dim in [0, 1]:
                        diff = array.shape[dim] - INPUT_SHAPE[dim]
                        margin[dim] = diff // 2
                    array = array[
                        margin[0] : margin[0] + INPUT_SHAPE[0],
                        margin[1] : margin[1] + INPUT_SHAPE[1],
                    ]
                    label = int(fpath.split("/")[-2].split(".")[0])
                    try:
                        assert array.shape[2] == 3
                        x_train.append(array)
                        y_train.append(label)
                    except (IndexError, AssertionError) as ex:
                        print(f"{fpath} failed shape check")
    return np.array(x_train, dtype=np.uint8), np.array(y_train, dtype=np.uint8)


ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "hidden_size": 768,
}


def interpret_image_size(image_size_arg):
    """Process the image_size argument whether a tuple or int."""
    if isinstance(image_size_arg, int):
        return (image_size_arg, image_size_arg)
    if (
        isinstance(image_size_arg, tuple)
        and len(image_size_arg) == 2
        and all(map(lambda v: isinstance(v, int), image_size_arg))
    ):
        return image_size_arg
    raise ValueError(
        f"The image_size argument must be a tuple of 2 integers or a single integer. Received: {image_size_arg}"
    )


@tf.keras.utils.register_keras_serializable()
class ClassToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert len(input_shape) == 3, f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = tf.keras.layers.Dense(hidden_size, name="query")
        self.key_dense = tf.keras.layers.Dense(hidden_size, name="key")
        self.value_dense = tf.keras.layers.Dense(hidden_size, name="value")
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# pylint: disable=too-many-instance-attributes
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                ),
                tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x, approximate=False))
                if hasattr(tf.keras.activations, "gelu")
                else tf.keras.layers.Lambda(lambda x: tfa.activations.gelu(x, approximate=False)),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_0")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_2")
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_model(
    image_size,
    patch_size,
    num_layers,
    hidden_size,
    num_heads,
    name,
    mlp_dim,
    classes,
    dropout=0.1,
    activation="linear",
    include_top=True,
    representation_size=None,
):
    """Build a ViT model.

    Args:
        image_size: The size of input images.
        patch_size: The size of each patch (must fit evenly in image_size)
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        num_layers: The number of transformer layers to use.
        hidden_size: The number of filters to use
        num_heads: The number of transformer heads
        mlp_dim: The number of dimensions for the MLP output in the transformers.
        dropout_rate: fraction of the units to drop for dense layers.
        activation: The activation to use for the final layer.
        include_top: Whether to include the final classification layer. If not,
            the output will have dimensions (batch_size, hidden_size).
        representation_size: The size of the representation prior to the
            classification layer. If None, no Dense layer is inserted.
    """
    image_size_tuple = interpret_image_size(image_size)
    assert (image_size_tuple[0] % patch_size == 0) and (
        image_size_tuple[1] % patch_size == 0
    ), "image_size must be a multiple of patch_size"
    x = tf.keras.layers.Input(shape=(image_size_tuple[0], image_size_tuple[1], 3))
    y = tf.keras.layers.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
    )(x)
    y = tf.keras.layers.Reshape((y.shape[1] * y.shape[2], hidden_size))(y)
    y = ClassToken(name="class_token")(y)
    y = AddPositionEmbs(name="Transformer/posembed_input")(y)
    for n in range(num_layers):
        y, _ = TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
        )(y)
    y = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")(y)
    y = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(y)
    if representation_size is not None:
        y = tf.keras.layers.Dense(representation_size, name="pre_logits", activation="tanh")(y)
    if include_top:
        y = tf.keras.layers.Dense(classes, name="head", activation=activation)(y)
    return tf.keras.models.Model(inputs=x, outputs=y, name=name)


def vit_b16(
    image_size: (224, 224),
    classes=1000,
    activation="linear",
    include_top=True,
):
    """Build ViT-B16. All arguments passed to build_model."""
    model = build_model(
        **ConfigDict,
        name="vit-b16",
        patch_size=16,
        image_size=image_size,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768,
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configure the VIT Training through Hyperparameters"
    )
    parser.add_argument(
        "--NUM_CLASSES", type=int, default=257, help="Number of classfication categories"
    )
    parser.add_argument(
        "--INPUT_SHAPE", type=int, nargs="+", default=[224, 224, 3], help="Shape of input to VIT"
    )
    parser.add_argument("--BATCH_SIZE", type=int, help="Batch Size to use with the Hardware")
    parser.add_argument(
        "--LEARNING_RATE", type=float, default=0.001, help="Learning rate to use for the Optimizer"
    )
    parser.add_argument(
        "--WEIGHT_DECAY", type=float, default=0.0001, help="Weight decay to use for the Optimizer"
    )
    parser.add_argument(
        "--EPOCHS", type=int, default=1, help="Number of times to loop over the data"
    )
    args, unused = parser.parse_known_args()

    args.INPUT_SHAPE = tuple(args.INPUT_SHAPE)
    print(f"Training on Images of size {args.INPUT_SHAPE}")

    x_train, y_train = load_dataset(args.INPUT_SHAPE, args.NUM_CLASSES)
    x_train = normalize(x_train)
    print(f"Training on dataset size {x_train.shape}")

    with strategy.scope():
        model = vit_b16(image_size=tuple(args.INPUT_SHAPE[:2]), classes=args.NUM_CLASSES)
        model.compile(
            optimizer=tfa.optimizers.AdamW(
                learning_rate=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY
            ),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )
        model.fit(x_train, y_train, epochs=args.EPOCHS, batch_size=args.BATCH_SIZE, verbose=2)
