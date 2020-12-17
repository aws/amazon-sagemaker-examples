# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tensorflow as tf

# Import SMDataParallel TensorFlow2 Modules
import smdistributed.dataparallel.tensorflow as dist

import os

tf.random.set_seed(42)


# SMDataParallel: Initialize
dist.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    # SMDataParallel: Pin GPUs to a single SMDataParallel process [use SMDataParallel local_rank() API]
    tf.config.experimental.set_visible_devices(gpus[dist.local_rank()], 'GPU')

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % dist.rank())

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(128)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
loss = tf.losses.SparseCategoricalCrossentropy()

# SMDataParallel: dist.size()
# LR for 8 node run : 0.000125
# LR for single node run : 0.001
opt = tf.optimizers.Adam(0.000125 * dist.size())

checkpoint_dir = os.environ['SM_MODEL_DIR']

checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)


@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    # SMDataParallel: Wrap tf.GradientTape with SMDataParallel's DistributedGradientTape
    tape = dist.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

    if first_batch:
       # SMDataParallel: Broadcast model and optimizer variables
       dist.broadcast_variables(mnist_model.variables, root_rank=0)
       dist.broadcast_variables(opt.variables(), root_rank=0)
    
    # SMDataParallel: all_reduce call
    loss_value = dist.oob_allreduce(loss_value) # Average the loss across workers
    return loss_value


for batch, (images, labels) in enumerate(dataset.take(10000 // dist.size())):
    loss_value = training_step(images, labels, batch == 0)

    if batch % 50 == 0 and dist.rank() == 0:
        print('Step #%d\tLoss: %.6f' % (batch, loss_value))

# SMDataParallel: Save checkpoints only from master node.
if dist.rank() == 0:
    mnist_model.save(os.path.join(checkpoint_dir, '1'))
