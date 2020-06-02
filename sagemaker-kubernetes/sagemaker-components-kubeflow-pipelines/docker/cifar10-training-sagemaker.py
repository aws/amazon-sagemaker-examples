import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from model_def import get_custom_model
import time
import argparse
import os
import re

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10

# Copy inference pre/post-processing script so that it'll be included in the model package
os.system('mkdir /opt/ml/model/code')
os.system('cp inference.py /opt/ml/model/code')
os.system('cp requirements.txt /opt/ml/model/code')

def single_example_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)
    
    image = train_preprocess_fn(image)
    label = tf.one_hot(label, NUM_CLASSES)
    
    return image, label

def train_preprocess_fn(image):

    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [HEIGHT, WIDTH, DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
    return image

def get_dataset(filenames, batch_size):
    """Read the images and labels from 'filenames'."""
    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(filenames).repeat().shuffle(10000)

    # Parse records.
    dataset = dataset.map(single_example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def get_model(model_type, input_shape, learning_rate, weight_decay, optimizer, momentum):
    input_tensor = Input(shape=input_shape)
    if model_type == 'resnet':
        base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                          weights='imagenet',
                                                          input_tensor=input_tensor,
                                                          input_shape=input_shape,
                                                          classes=None)
        x = Flatten()(base_model.output)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    elif model_type == 'vgg':
        base_model = keras.applications.vgg19.VGG19(include_top=False,
                                                          weights=None,
                                                          input_tensor=input_tensor,
                                                          input_shape=input_shape,
                                                          classes=None)
        x = Flatten()(base_model.output)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    else:
        model = get_custom_model(input_shape, learning_rate, weight_decay, optimizer, momentum)
        
    return model

def main(args):
    # Hyper-parameters
    epochs       = args.epochs
    lr           = args.learning_rate
    batch_size   = args.batch_size
    momentum     = args.momentum
    weight_decay = args.weight_decay
    optimizer    = args.optimizer
    model_type   = args.model_type

    # SageMaker options
    training_dir   = args.training
    validation_dir = args.validation
    eval_dir       = args.eval

    train_dataset = get_dataset(training_dir+'/train.tfrecords',  batch_size)
    val_dataset   = get_dataset(validation_dir+'/validation.tfrecords', batch_size)
    eval_dataset  = get_dataset(eval_dir+'/eval.tfrecords', batch_size)
    
    input_shape = (HEIGHT, WIDTH, DEPTH)
    
    # Load model
    model = get_model(model_type, input_shape, lr, weight_decay, optimizer, momentum)
            
    # Optimizer
    if optimizer.lower() == 'sgd':
        opt = SGD(lr=lr, decay=weight_decay, momentum=momentum)
    else:
        opt = Adam(lr=lr, decay=weight_decay)

    # Compile model
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train model
    history = model.fit(train_dataset, steps_per_epoch=40000 // batch_size,
                        validation_data=val_dataset, 
                        validation_steps=10000 // batch_size,
                        epochs=epochs)
    
    # Evaluate model performance
    score = model.evaluate(eval_dataset, steps=10000 // batch_size, verbose=1)
    print('Test loss    :', score[0])
    print('Test accuracy:', score[1])
    
    # Save model to model directory
    model.save(f'{os.environ["SM_MODEL_DIR"]}/{time.strftime("%m%d%H%M%S", time.gmtime())}', save_format='tf')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Hyper-parameters
    parser.add_argument('--epochs',        type=int,   default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size',    type=int,   default=128)
    parser.add_argument('--weight-decay',  type=float, default=2e-4)
    parser.add_argument('--momentum',      type=float, default='0.9')
    parser.add_argument('--optimizer',     type=str,   default='sgd')
    parser.add_argument('--model-type',    type=str,   default='resnet')

    # SageMaker parameters
    parser.add_argument('--model_dir',        type=str,   default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training',         type=str,   default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation',       type=str,   default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--eval',             type=str,   default=os.environ['SM_CHANNEL_EVAL'])
    
    args = parser.parse_args()
    main(args)
