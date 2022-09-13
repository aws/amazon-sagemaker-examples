from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow_addons as tfa
import tensorflow as tf
import os
import horovod.tensorflow.keras as hvd

# parse TFRecord
def parse_image_function(example_proto):
    image_feature_description = {'image': tf.io.FixedLenFeature([], tf.string),
                                 'label': tf.io.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(
        example_proto, image_feature_description)
    image = tf.io.decode_raw(features['image'], tf.uint8)
    image.set_shape([3 * 32 * 32])
    image = tf.reshape(image, [32, 32, 3])
    label = tf.cast(features['label'], tf.int32)
    return image, label


# dilation filter
def dilate(image, label):
    dilateFilter = tf.zeros([3, 3, 3], tf.uint8)
    image = tf.expand_dims(image, 0)
    image = tf.nn.dilation2d(
        image, dilateFilter, strides=[1, 1, 1, 1],
        dilations=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC')
    image = tf.squeeze(image)
    return image, label
# blur filter


def blur(image, label):
    image = tfa.image.gaussian_filter2d(image=image,
                                        filter_shape=(11, 11), sigma=0.8)
    return image, label

# rescale filter
def rescale(image, label):
    image = preprocessing.Rescaling(1.0 / 255)(image)
    return image, label


# augmentation filters
def augment(image, label):
    data_augmentation = tf.keras.Sequential(
        [preprocessing.RandomFlip("horizontal"),
         preprocessing.RandomRotation(0.1),
         preprocessing.RandomZoom(0.1)])
    image = data_augmentation(image)
    return image, label


def get_dataset(training_dir : str, batch_size : int, use_tf_data_service : bool, dispatcher_host : str):
    autotune = tf.data.experimental.AUTOTUNE
    options = tf.data.Options()
    options.experimental_deterministic = False
    records = tf.data.Dataset.list_files(
        training_dir+'/*', shuffle=True).with_options(options)
    ds = tf.data.TFRecordDataset(records, num_parallel_reads=autotune).repeat()
    
    ds = ds.map(parse_image_function, num_parallel_calls=autotune)
    ds = ds.map(dilate, num_parallel_calls=autotune)
    ds = ds.map(blur, num_parallel_calls=autotune)
    ds = ds.map(rescale,num_parallel_calls=autotune)

    ds = ds.map(augment, num_parallel_calls=autotune)
    ds = ds.batch(batch_size)

    if use_tf_data_service:
        ds = ds.apply(tf.data.experimental.service.distribute(
            processing_mode="parallel_epochs",
            service=f'grpc://{dispatcher_host}:6000',),
        )

    #ds = ds.take(1).cache().repeat()
    ds = ds.prefetch(autotune)
    return ds


"This function read mode command line argument"
def read_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_data_mode', type=str, default='local',
                        help="'service' distributed dataset using tf.data.service. 'local' use standard tf.data")
    parser.add_argument('--steps_per_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument("--n_gpus", type=str,
                        default=os.environ.get("SM_NUM_GPUS"))
    parser.add_argument("--training_dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--dispatcher_host", type=str)
    parser.add_argument("--num_of_data_workers", type=int, default=1)
    parser.add_argument("--output_data_dir", type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--checkpoint-path",type=str,default="/opt/ml/checkpoints",help="Path where checkpoints are saved.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = read_args()
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(str(gpus))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f'hvd.local_rank() {hvd.local_rank()}')
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
    model = ResNet50(weights=None,
                     input_shape=(32, 32, 3),
                     classes=10)
    
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.optimizers.Adam())
   # Horovod: adjust learning rate based on number of GPUs.
    scaled_lr = 0.001 * hvd.size()
    opt = tf.optimizers.Adam(scaled_lr)
    opt = hvd.DistributedOptimizer(
        opt, backward_passes_per_step=1, average_aggregated_gradients=True)
    
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                  optimizer=opt,
                  experimental_run_tf_function=False)

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
    ]
     # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        path = os.path.join(args.checkpoint_path, './checkpoint-{epoch}.h5')
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(path))

     # Horovod: write logs on worker 0.
    verbose = 1 if hvd.rank() == 0 else 0

    assert(args.tf_data_mode == 'local' or args.tf_data_mode == 'service')
    print(f'Running in {args.tf_data_mode} tf_data_mode.')
    dataset = get_dataset(args.training_dir, batch_size = args.batch_size, use_tf_data_service=(args.tf_data_mode == 'service'), dispatcher_host = args.dispatcher_host)
    
    model.fit(  dataset, 
                steps_per_epoch=args.steps_per_epoch,
                callbacks=callbacks,
                epochs=args.epochs, 
                verbose=2,)

    if hvd.rank() == 0:
        model.save(os.path.join(args.model_dir, '000000001'), 'my_model.h5')