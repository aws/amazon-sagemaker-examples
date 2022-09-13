import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers.experimental import preprocessing

DISPATCHER_HOST='localhost'

# parse TFRecord+
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

'this function loads mnist as tf.data.Dataset'
def load_mnist():
    import numpy as np
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_train = np.expand_dims(x_train, -1)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    return train_dataset

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

    ds = ds.prefetch(autotune)
    return ds

"This function read mode command line argument"
def read_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='local',
                        help='Mode to run the script: local or service')
    parser.add_argument("--training_dir", type=str, default='data')
    parser.add_argument('--batch_size', type=int, default = 2048)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args()
    return args

def shutdown_tf_data_service():
    SHUTDOWN_PORT = 16000
    print('Shutting down tf.data.service dispatcher via port {}'.format(SHUTDOWN_PORT))
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', SHUTDOWN_PORT))
    s.close()

if __name__ == "__main__":  
    args = read_args()
    mode = args.mode
    model = ResNet50(weights=None,
                     input_shape=(32, 32, 3),
                     classes=10)
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.optimizers.Adam())
    
    assert(mode == 'local' or mode == 'service')
    print(f'Running in {mode} mode.')
    
    dataset = get_dataset(args.training_dir, batch_size = 1024, use_tf_data_service=(mode == 'service'), dispatcher_host = DISPATCHER_HOST)
    
    model.fit(dataset, steps_per_epoch=1, epochs=2, verbose=2)

    model.save(os.path.join(args.model_dir, '000000001'), 'my_model.h5')

    if mode == 'service':
        shutdown_tf_data_service()