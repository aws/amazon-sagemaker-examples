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
    # Transpose
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