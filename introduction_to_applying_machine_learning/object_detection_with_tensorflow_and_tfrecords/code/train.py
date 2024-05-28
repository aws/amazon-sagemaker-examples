# Import necessary libraries and modules
import argparse
import json
import logging
import os
import sys
import subprocess
# Import TensorFlow and related modules
import tensorflow as tf
from tensorflow import data as tf_data
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import  ModelCheckpoint
import keras
import keras_cv
from keras_cv import bounding_box

# Set TensorFlow logging level to suppress unnecessary output
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.get_logger().setLevel(1)

# Enable eager execution (optional)
#tf.compat.v1.enable_eager_execution

# Configure GPU settings for TensorFlow
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ["KERAS_BACKEND"] = "tensorflow"

# Set logging levels
logging.getLogger().setLevel(logging.INFO)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Create directories for storing model and checkpoints
subprocess.run(['mkdir','/opt/ml/model/code'])
subprocess.run(['mkdir', '/opt/ml/checkpoints'])
subprocess.run(['cp', 'requirements.txt', '/opt/ml/model/code'])

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Set up logging to a file in the SageMaker metrics directory
if "SAGEMAKER_METRICS_DIRECTORY" in os.environ:
    log_file_handler = logging.FileHandler(
        os.path.join(os.environ["SAGEMAKER_METRICS_DIRECTORY"], "metrics.json")
    )
    formatter = logging.Formatter(
        "{'time':'%(asctime)s', 'name': '%(name)s', \
        'level': '%(levelname)s', 'message': '%(message)s'}",
        style="%",
    )
    log_file_handler.setFormatter(formatter)
    logger.addHandler(log_file_handler)


def parse_args():
    """
    Parse command-line arguments and return the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--global_clipnorm', type=float, default=0.3)
    parser.add_argument('--finetune', type=bool, default=True)
    parser.add_argument('--region', type=str, default='us-west-2')
    parser.add_argument('--env_parameters', type=str, default=os.environ['SM_TRAINING_ENV'])
    

    #GPU hyperparameters
    #parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
        
    # Data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--eval', type=str, default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--train_samples', type=int, default=1000)
    parser.add_argument('--eval_samples', type=int, default=100)
    
    # Model directory and checkpoint path
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument("--checkpoint_path",type=str,default="/opt/ml/checkpoints",help="Path where checkpoints will be saved.")
    
    return parser.parse_known_args()


# Filter out overlapping bounding boxes based on intersection over union (IoU) thresholds
prediction_decoder = keras_cv.layers.NonMaxSuppression(
    bounding_box_format="xyxy",
    from_logits=True,
    # Decrease the required threshold to make predictions get pruned out
    iou_threshold=1.0,
    # Tune confidence threshold for predictions to pass NMS
    confidence_threshold=0.0,
)

def get_model():
    """
    Get the RetinaNet model with specified configuration.
    """
    if args.finetune:
        model = keras_cv.models.RetinaNet.from_preset(
            "retinanet_resnet50_pascalvoc", bounding_box_format="xyxy",
            #prediction_decoder=prediction_decoder,
            num_classes=20,
            load_weights=True)
    else:
        model = keras_cv.models.RetinaNet.from_preset(
            "retinanet_resnet50_pascalvoc", bounding_box_format="xyxy",
            #prediction_decoder=prediction_decoder,
            num_classes=20,
            load_weights=False)
    return model
 


def parse_tfrecord_fn(example):
    """
    Parse a TFRecord example and return a dictionary of features.
    """
    feature_description = {
        "height": tf.io.FixedLenFeature((), tf.int64),
        "width": tf.io.FixedLenFeature((), tf.int64),
        "filename":tf.io.FixedLenFeature((),tf.string),
        "image": tf.io.FixedLenFeature((), tf.string),
        "object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "object/bbox/ymin":tf.io.VarLenFeature(tf.float32),
        "object/bbox/ymax":tf.io.VarLenFeature(tf.float32),
        "object/text":tf.io.VarLenFeature(tf.string),
        "object/label": tf.io.VarLenFeature(tf.int64),
    }

    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.cast(tf.io.decode_jpeg(example["image"], channels=3),tf.float32)
    example["filename"] = tf.cast(example["filename"], tf.string)
    example["object/bbox/xmin"] = tf.sparse.to_dense(example["object/bbox/xmin"])
    example["object/bbox/xmax"] = tf.sparse.to_dense(example["object/bbox/xmax"])
    example["object/bbox/ymin"] = tf.sparse.to_dense(example["object/bbox/ymin"])
    example["object/bbox/ymax"] = tf.sparse.to_dense(example["object/bbox/ymax"])
    example["object/text"] = tf.sparse.to_dense(example["object/text"])
    example["object/label"] = tf.sparse.to_dense(example["object/label"])
    example["object/bbox"] = tf.stack([example["object/bbox/xmin"],example["object/bbox/ymin"],example["object/bbox/xmax"],example["object/bbox/ymax"]],axis=1)
    
    return example

def prepare_sample(inputs):
    """
    Prepare sample data for training/evaluation.
    """
    image = inputs['image']
    boxes = inputs["object/bbox"]
    bounding_boxes = {
        "classes": inputs["object/label"],
        "boxes": boxes,
    }
    return {"images": image, "bounding_boxes": bounding_boxes}



def dict_to_tuple(inputs):
    """
    Convert dictionary inputs to a tuple of images and bounding boxes.
    """
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32)


    
def get_train_data(train_dir,batch_size,epochs):
    """
    Load and preprocess training data.
    """
    filenames=[os.path.join(train_dir, f) for f in os.listdir(f"{train_dir}/")]   
 
    def apply_pipeline(inputs):
        inputs["images"] = pipeline(inputs["images"])
        return inputs

    augmenter = keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
            keras_cv.layers.JitteredResize(
                target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xyxy"
            ),
        ]
    )
    
    pipeline = keras_cv.layers.RandomAugmentationPipeline(
        layers=[keras_cv.layers.GridMask(ratio_factor=(0, 0.3),),keras_cv.layers.RandomColorDegeneration(.5),keras_cv.layers.RandomSaturation(.8),],
        augmentations_per_image=3,
    )
    
    dataset = tf.data.TFRecordDataset(filenames,num_parallel_reads=tf.data.experimental.AUTOTUNE)
    BATCH_SIZE=batch_size
    raw_dataset_sample = dataset.map(parse_tfrecord_fn)
    
    parsed_dataset_sample = raw_dataset_sample.map(
        lambda x: prepare_sample(x)
    )
    
    dataset_ds = parsed_dataset_sample.shuffle(BATCH_SIZE * 4)
    dataset_ds = dataset_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    dataset_ds = dataset_ds.map(apply_pipeline, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_ds = dataset_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
    
    train_ds = dataset_ds.map(dict_to_tuple, num_parallel_calls=tf_data.AUTOTUNE)
    #train_ds = train_ds.cache()
    train_ds=train_ds.repeat(epochs)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds


def get_val_data(test_dir,batch_size):
    """
    Load and preprocess validation data.
    """
    filenames=[os.path.join(train_dir, f) for f in os.listdir(f"{test_dir}/")]
    #print("Number of validation samples: ",sum(1 for _ in tf.data.TFRecordDataset(filenames)))
    
    dataset = tf.data.TFRecordDataset(filenames,num_parallel_reads=tf.data.experimental.AUTOTUNE)
    BATCH_SIZE=batch_size
    
    raw_dataset_eval = dataset.map(parse_tfrecord_fn)
    parsed_dataset_sample = raw_dataset_eval.map(
        lambda x: prepare_sample(x)
    )
    
    val_ds = parsed_dataset_sample.shuffle(BATCH_SIZE * 4)
    val_ds = val_ds.ragged_batch(BATCH_SIZE)
    
    inference_resizing = keras_cv.layers.Resizing(
        640, 640, pad_to_aspect_ratio=True, bounding_box_format="xyxy"
    )
    val_ds = val_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf_data.AUTOTUNE)
    
    #val_ds=val_ds.repeat()
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    return val_ds
    
if __name__ == "__main__":    
    args, _ = parse_args()
        
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    global_clipnorm=args.global_clipnorm
    model_dir=args.model_dir
    checkpoint_path=args.checkpoint_path
    region=args.region
    train_samples=args.train_samples
    eval_samples=args.eval_samples
    sm_training_env=args.env_parameters
    
    sm_training_env = json.loads(sm_training_env)
    job_name=sm_training_env['job_name']
    print(sm_training_env['job_name'])
    
    train_dir=args.train
    val_dir=args.eval
    
    envs = dict(os.environ)
    sm_training_env = envs.get('SM_TRAINING_ENV')
    
    print(f"train: {train_dir}")
    print(f"val: {val_dir}")
    
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"number of gpus: {gpus}")
    
    # Create a MirroredStrategy for multiple GPU
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    train_ds= get_train_data(train_dir,batch_size,epochs)
    val_ds = get_val_data(val_dir,batch_size)
    
    # Open a strategy scope for distributed training
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope
        model = get_model()
        optimizer =keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            clipnorm=global_clipnorm,
            global_clipnorm=None,
            use_ema=True,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            name="SGD",
        )
        # Compile the model with specified losses and metrics
        model.compile(classification_loss="focal",
                        box_loss="smoothl1",
                        optimizer=optimizer,
                        metrics=[keras_cv.metrics.BoxCOCOMetrics(bounding_box_format="xyxy",evaluate_freq=1e9,)])    
    size=1
    
    # Prepare callbacks
    callbacks = []
    callbacks.append(keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
    callbacks.append(ModelCheckpoint(args.checkpoint_path + '/checkpoint-{epoch}.keras'))
    
    # Train the model
    model.fit(train_ds,
              # Run for 10-35~ epochs to achieve good scores.
              epochs=epochs,
              steps_per_epoch=10,#(train_samples // args.batch_size)//size,
              callbacks=callbacks,
              shuffle=True,
              )
    
    # Save the model
    # A version number is needed for the serving container
    # to load the model
    version = "1"
    ckpt_dir = os.path.join(model_dir, version)
    print(f'saving model on {ckpt_dir}')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    tf.saved_model.save(model,ckpt_dir)
    model.save(f"{ckpt_dir}/model.keras")
    model.save_weights(f"{ckpt_dir}/model.weights.h5")