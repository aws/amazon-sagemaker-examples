import os, time
import errno
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# SMP: Import TF2.x API
import smdistributed.modelparallel.tensorflow as smp

tf.random.set_seed(1234)

# SMP: Initialize
smp.init()

cache_dir = os.path.join(os.path.expanduser("~"), ".keras", "datasets")
if not os.path.exists(cache_dir):
    try:
        os.mkdir(cache_dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
            pass
        else:
            raise

# Download and load MNIST dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
    "MNIST-data-%d" % smp.rank()
)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# SMP: If needed, seed the shuffle with smp.dp_rank(), and drop_remainder
# in batching to make sure batch size is always divisible by number of microbatches
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000, seed=smp.dp_rank())
    .batch(256, drop_remainder=True)
)
test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .shuffle(10000, seed=smp.dp_rank())
    .batch(256, drop_remainder=True)
)

# SMP: Define smp.DistributedModel the same way as Keras sub-classing API 
class MyModel(smp.DistributedModel):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.dense1 = Dense(128)
        self.dense2 = Dense(10)

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
test_loss = tf.keras.metrics.Mean(name="test_loss")
train_loss = tf.keras.metrics.Mean(name="train_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

checkpoint_directory = "/opt/ml/checkpoints"

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

ckpt_manager = smp.CheckpointManager(checkpoint, checkpoint_directory)

def aws_s3_sync(source, destination):
    """aws s3 sync in quiet mode and time profile"""
    import time, subprocess
    cmd = ["aws", "s3", "sync", "--quiet", source, destination]
    print(f"Syncing files from {source} to {destination}")
    start_time = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    end_time = time.time()
    print("Time Taken to Sync: ", (end_time-start_time))
    return

def sync_local_checkpoints_to_s3(local_path="/opt/ml/checkpoints", s3_path=os.path.dirname(os.path.dirname(os.getenv('SM_MODULE_DIR', '')))+'/checkpoints'):
    """ sample function to sync checkpoints from local path to s3 """

    import boto3, botocore
    #check if local path exists
    if not os.path.exists(local_path):
        raise RuntimeError("Provided local path {local_path} does not exist. Please check")

    #check if s3 bucket exists
    s3 = boto3.resource('s3')
    if 's3://' not in s3_path:
        raise ValueError("Provided s3 path {s3_path} is not valid. Please check")

    s3_bucket = s3_path.replace('s3://','').split('/')[0]
    print(f"S3 Bucket: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            raise RuntimeError('S3 bucket does not exist. Please check')
    aws_s3_sync(local_path, s3_path)
    return

def sync_s3_checkpoints_to_local(local_path="/opt/ml/checkpoints", s3_path=os.path.dirname(os.path.dirname(os.getenv('SM_MODULE_DIR', '')))+'/checkpoints'):
    """ sample function to sync checkpoints from s3 to local path """

    import boto3, botocore
    #creat if local path does not exists
    if not os.path.exists(local_path):
        print(f"Provided local path {local_path} does not exist. Creating...")
        try:
            os.makedirs(local_path)
        except Exception as e:
            raise RuntimeError(f"failed to create {local_path}")

    #check if s3 bucket exists
    s3 = boto3.resource('s3')
    if 's3://' not in s3_path:
        raise ValueError("Provided s3 path {s3_path} is not valid. Please check")

    s3_bucket = s3_path.replace('s3://','').split('/')[0]
    print(f"S3 Bucket: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            raise RuntimeError('S3 bucket does not exist. Please check')
    aws_s3_sync(s3_path, local_path)
    return

# SMP: Define smp.step. Return any tensors needed outside
@smp.step
def get_grads(images, labels):
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)

    grads = optimizer.get_gradients(loss, model.trainable_variables)
    return grads, loss, predictions


@tf.function
def train_step(images, labels):
    gradients, loss, predictions = get_grads(images, labels)

    # SMP: Accumulate the gradients across microbatches
    gradients = [g.accumulate() for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # SMP: Average the loss across microbatches
    train_loss(loss.reduce_mean())

    # SMP: Merge predictions across microbatches
    train_accuracy(labels, predictions.merge())
    return loss.reduce_mean()


# SMP: Define the smp.step for evaluation. Optionally specify an input signature.
@smp.step(
    input_signature=[
        tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float64),
        tf.TensorSpec(shape=[None], dtype=tf.uint8),
    ]
)
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    return t_loss


for epoch in range(5):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print(
        "Epoch {}, Loss: {}, Accuracy: {}, Test loss {}, test accuracy {}".format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result(),
            test_loss.result(),
            test_accuracy.result(),
        )
    )

ckpt_manager.save()
smp.barrier()

if smp.local_rank() == 0:
    print("Starting sync")
    base_s3_path = os.path.dirname(os.path.dirname(os.getenv('SM_MODULE_DIR', '')))
    curr_host = os.getenv('SM_CURRENT_HOST')
    full_s3_path = f'{base_s3_path}/checkpoints/{curr_host}/'
    sync_local_checkpoints_to_s3(s3_path=full_s3_path)
    print("Finished sync")
