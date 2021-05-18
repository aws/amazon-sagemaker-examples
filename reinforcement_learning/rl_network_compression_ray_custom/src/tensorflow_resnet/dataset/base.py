from . import cifar10


class Cifar10:
    height = cifar10._HEIGHT
    width = cifar10._WIDTH
    num_channels = cifar10._NUM_CHANNELS
    data_dir = cifar10._DATA_DIR
    input_fn = cifar10.input_fn
    num_classes = cifar10.NUM_CLASSES
    num_images = cifar10.NUM_IMAGES
