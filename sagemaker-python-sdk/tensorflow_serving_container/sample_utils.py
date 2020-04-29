import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2


def tfhub_to_savedmodel(model_name, export_path,
                        uri_pattern='https://tfhub.dev/google/imagenet/{}/classification/2'):
    """Download a model from TensorFlow Hub, add inputs and outputs
    suitable for serving inference requests, and export the resulting
    graph as a SavedModel. This function should work for most
    image classification model on TensorFlow Hub.
    
    Args:
        model_name (str): The model name (e.g. mobilenet_v2_140_224)
        export_path (str): The exported model will be saved at <export_path>/<model_name>
        uri_pattern (str): Optional.  The model name is combined with this 
            pattern to form a TensorFlow Hub uri. The default value works for MobileNetV2, 
            but a different pattern may be needed for other models.
        
    Returns:
        str: The path to the exported SavedModel (including model_name and version).
    """

    # the model will output the topk predicted classes and probabilities
    topk = 3 
    
    model_path = '{}/{}/00000001'.format(export_path, model_name)
    tfhub_uri = uri_pattern.format(model_name)

    with tf.Session(graph=tf.Graph()) as sess:
        module = hub.Module(tfhub_uri) 
        input_params = module.get_input_info_dict()                
        dtype = input_params['images'].dtype
        shape = input_params['images'].get_shape()

        # define the model inputs
        inputs = {'images': tf.placeholder(dtype, shape, 'images')}

        # define the model outputs
        # we want the class ids and probabilities for the top 3 classes
        logits = module(inputs['images'])
        softmax = tf.nn.softmax(logits, name=None)
        probs, classes = tf.nn.top_k(softmax, k=topk, sorted=True, name=None)
        outputs = {
            'classes': classes,
            'probabilities': probs
        }

        # export the model
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        tf.saved_model.simple_save(
            sess,
            model_path,
            inputs=inputs,
            outputs=outputs)  

    return model_path


def image_file_to_tensor(path):
    """Reads an image file and coverts it to a tensor (ndarray). 
    
    No resizing or cropping is done, so the image dimensions must match
    the model input shape (224x224 for the mobilenet_v2_140_224 model).
    
    Args: 
        path (str): The file name or path to the image file.
    """
    
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image)
    image = cv2.normalize(image.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    image = np.expand_dims(image, axis=0)
    return image


def add_imagenet_labels(prediction_result):
    """Add imagenet class labels to the prediction result. The
    prediction_result argument will be modified in place.
    """

    # read the labels from a file
    labels = []
    with open('labels.txt', 'r') as f:
        labels = [l.strip() for l in f]
    
    # add labels to the result dict
    for pred in prediction_result['predictions']:
        prediction_labels = [labels[x - 1] for x in pred['classes']]
        pred['labels'] = prediction_labels

        
def print_probabilities_and_labels(labelled_result):
    """Print the labelled results."
    """
        
    for pred in labelled_result['predictions']:
        for i in range(0, len(pred['labels'])):
            print('{:1.7f} {}'.format(
                pred['probabilities'][i],
                pred['labels'][i],
            ))
        print()
        
