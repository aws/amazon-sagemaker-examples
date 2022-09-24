import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers.experimental import preprocessing

DISPATCHER_HOST='localhost'

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

# This function generates a dataset consisting 32x32x3 random images
# And a corresponding random label representing 10 different classes.
# As this dataset is randomly generated, you should not expect the model
# to converge in a meaningful way, it doesn't matter as our intent is 
# only to measure data pipeline and DNN optimization throughput
def generate_artificial_dataset():
    import numpy as np
    x_train = np.random.randint(0, 255, size=(32000, 32, 32, 3), dtype=np.uint8)
    y_train = np.random.randint(0, 10, size=(32000,1))
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    return train_dataset

def get_dataset(training_dir : str, batch_size : int, use_tf_data_service : bool, dispatcher_host : str):
    autotune = tf.data.experimental.AUTOTUNE
    options = tf.data.Options()
    options.experimental_deterministic = False
    
    ds = generate_artificial_dataset().shuffle(10000).repeat()
    
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