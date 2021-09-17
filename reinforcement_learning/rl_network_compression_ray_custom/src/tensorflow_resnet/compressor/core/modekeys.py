import tensorflow as tf


class ModeKeys(tf.estimator.ModeKeys):
    """Extending for a compress mode"""

    COMPRESS = "compress"
    REFERENCE = "reference"
