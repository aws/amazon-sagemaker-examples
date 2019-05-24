import tensorflow as tf


def get_model():
    
    inputs = tf.keras.Input(shape=(13,))
    hidden_1 = tf.keras.layers.Dense(13, activation='tanh')(inputs)
    hidden_2 = tf.keras.layers.Dense(6, activation='sigmoid')(hidden_1)
    outputs = tf.keras.layers.Dense(1)(hidden_2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
 