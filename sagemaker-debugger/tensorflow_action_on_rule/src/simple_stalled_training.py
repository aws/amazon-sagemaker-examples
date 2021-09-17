"""
This script is a simple training script with the Tensorflow MonitoredSession interface 
with 4 additional lines of code at the end to force the training job stalled.
It is designed to be used with SageMaker Debugger in an official SageMaker Framework container (i.e. AWS Deep Learning Container).
You can also manually register the Debugger hook object if your training script requires frameworks and versions
that are not supported by the Deep Learning Containers. The manual hook registration code lines are provided
in this training script as commented lines. If you want to explore, uncomment the lines and run it on
other frameworks and containers.
For more information, refer to https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html
"""

# Standard Library
import argparse
import random

# Third Party
import numpy as np
import tensorflow.compat.v1 as tf

# First Party
# import smdebug.tensorflow as smd


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, help="S3 path for the model")
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.001)
parser.add_argument("--steps", type=int, help="Number of steps to run", default=100)
parser.add_argument("--scale", type=float, help="Scaling factor for inputs", default=1.0)
parser.add_argument("--random_seed", type=bool, default=False)
args = parser.parse_args()

# these random seeds are only intended for test purpose.
# for now, 2,2,12 could promise no assert failure when running tests
# if you wish to change the number, notice that certain steps' tensor value may be capable of variation
if args.random_seed:
    tf.set_random_seed(2)
    np.random.seed(2)
    random.seed(12)

# Network definition
# Note the use of name scopes
with tf.name_scope("foobar"):
    x = tf.placeholder(shape=(None, 2), dtype=tf.float32)
    w = tf.Variable(initial_value=[[10.0], [10.0]], name="weight1")
with tf.name_scope("foobaz"):
    w0 = [[1], [1.0]]
    y = tf.matmul(x, w0)
loss = tf.reduce_mean((tf.matmul(x, w) - y) ** 2, name="loss")

# ========= Debugger Hook Registration =========
# Register a smdebug's TensorFlow session hook
# hook = smd.SessionHook.create_from_json_file()

# set up a "losses" collection using the hook
# hook.add_to_collection("losses", loss)
# ==============================================

global_step = tf.Variable(17, name="global_step", trainable=False)
increment_global_step_op = tf.assign(global_step, global_step + 1)

optimizer = tf.train.AdamOptimizer(args.lr)

# ========= Debugger Hook Registration =========
# Wrap the optimizer using the hook
# optimizer = hook.wrap_optimizer(optimizer)
# ==============================================

optimizer_op = optimizer.minimize(loss, global_step=increment_global_step_op)

sess = tf.train.MonitoredSession()

# ========= Debugger Hook Registration =========
# Pass the Debugger hook as a callback
# sess = tf.train.MonitoredSession(hooks=hook)
# ==============================================

# ========= Debugger Hook Registration =========
# Set the hook to TRAIN (train) mode
# hook.set_mode(smd.modes.TRAIN)
# ==============================================
for i in range(args.steps):
    x_ = np.random.random((10, 2)) * args.scale
    _loss, opt, gstep = sess.run([loss, optimizer_op, increment_global_step_op], {x: x_})
    print(f"Step={i}, Loss={_loss}")

# ========= Debugger Hook Registration =========
# Set the hook to EVAL (evaluation) mode
# hook.set_mode(smd.modes.EVAL)
# ==============================================
for i in range(args.steps):
    x_ = np.random.random((10, 2)) * args.scale
    sess.run([loss, increment_global_step_op], {x: x_})

# Force the training job to sleep for 10 minutes
import time

print("Sleeping for 10 minutes")
time.sleep(10 * 60)
print("Waking up and exiting")
