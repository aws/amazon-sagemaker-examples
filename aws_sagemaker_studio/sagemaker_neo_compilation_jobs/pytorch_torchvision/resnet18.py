def neo_preprocess(payload, content_type):
    import PIL.Image   # Training container doesn't have this package
    import logging
    import numpy as np
    import io

    logging.info('Invoking user-defined pre-processing function')

    if content_type != 'application/x-image':
        raise RuntimeError('Content type must be application/x-image')

    f = io.BytesIO(payload)
    # Load image and convert to RGB space
    image = PIL.Image.open(f).convert('RGB')
    # Resize
    image = np.asarray(image.resize((224, 224)))

    # Normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    image = (image/255- mean_vec)/stddev_vec

    # Transpose
    if len(image.shape) == 2:  # for greyscale image
        image = np.expand_dims(image, axis=2)
    
    image = np.rollaxis(image, axis=2, start=0)[np.newaxis, :]
    
    return image

def neo_postprocess(result):
    import logging
    import numpy as np
    import json

    logging.info('Invoking user-defined post-processing function')

    # Softmax (assumes batch size 1)
    result = np.squeeze(result)
    result_exp = np.exp(result - np.max(result))
    result = result_exp / np.sum(result_exp)

    response_body = json.dumps(result.tolist())
    content_type = 'application/json'

    return response_body, content_type