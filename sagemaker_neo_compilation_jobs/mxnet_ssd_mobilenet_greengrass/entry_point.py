def neo_preprocess(payload, content_type):
    import logging
    import numpy as np
    import io
    import PIL.Image
    def _read_input_shape(signature):
        shape = signature[-1]['shape']
        shape[0] = 1
        return shape

    def _transform_image(image, shape_info):
        # Fetch image size
        input_shape = _read_input_shape(shape_info)

        # Perform color conversion
        if input_shape[-3] == 3:
            # training input expected is 3 channel RGB
            image = image.convert('RGB')
        elif input_shape[-3] == 1:
            # training input expected is grayscale
            image = image.convert('L')
        else:
            # shouldn't get here
            raise RuntimeError('Wrong number of channels in input shape')

        # Resize
        image = np.asarray(image.resize((input_shape[-2], input_shape[-1])))

        # Normalize
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        image = (image/255- mean_vec)/stddev_vec

        # Transpose
        if len(image.shape) == 2:  # for greyscale image
            image = np.expand_dims(image, axis=2)
        image = np.rollaxis(image, axis=2, start=0)[np.newaxis, :]

        return image
    
    logging.info('Invoking user-defined pre-processing function')

    if content_type != 'image/jpeg':
        raise RuntimeError('Content type must be image/jpeg')
    
    shape_info = [{"shape":[1,3,512,512], "name":"data"}]
    f = io.BytesIO(payload)
    dtest = _transform_image(PIL.Image.open(f), shape_info)
    return {'data':dtest}

    

### NOTE: this function cannot use MXNet
def neo_postprocess(result):
    import logging
    import numpy as np
    import json

    logging.info('Invoking user-defined post-processing function')
 
    js = {'prediction':[],'instance':[]}
    for r in result:
        r = np.squeeze(r)
        js['instance'].append(r.tolist())
    idx, score, bbox = js['instance']
    bbox = np.asarray(bbox)/512
    res = np.hstack((np.column_stack((idx,score)),bbox))
    for r in res:
        js['prediction'].append(r.tolist())
    del js['instance']
    response_body = json.dumps(js)
    content_type = 'application/json'

    return response_body, content_type