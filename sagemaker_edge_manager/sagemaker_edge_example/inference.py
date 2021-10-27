import grpc
from PIL import Image
import agent_pb2
import agent_pb2_grpc
import os
import numpy as np
import json


model_path = os.environ['MODEL_PATH'] 
ml_model = os.environ['ML_MODEL']
image_path = os.environ['IMAGE_PATH']
capture_data = os.environ['CAPTURE_DATA'].lower() == "true" 
                    
agent_socket = 'unix:///tmp/aws.greengrass.SageMakerEdgeManager.sock'

agent_channel = grpc.insecure_channel(agent_socket, options=(('grpc.enable_http_proxy', 0),))

agent_client = agent_pb2_grpc.AgentStub(agent_channel)


def list_models():
    return agent_client.ListModels(agent_pb2.ListModelsRequest())


def list_model_tensors(models):
    return {
        model.name: {
            'inputs': model.input_tensor_metadatas,
            'outputs': model.output_tensor_metadatas
        }
        for model in list_models().models
    }


def load_model(model_name, model_path):
    load_request = agent_pb2.LoadModelRequest()
    load_request.url = model_path
    load_request.name = model_name
    return agent_client.LoadModel(load_request)


def unload_model(name):
    unload_request = agent_pb2.UnLoadModelRequest()
    unload_request.name = name
    return agent_client.UnLoadModel(unload_request)


def predict_image(model_name, image_path):
    image_tensor = agent_pb2.Tensor()
    im = Image.open(image_path)
    img = np.asarray(im)
    # Neo compiled model requires the array to be of shape (3, 244, 244)
    img = img.transpose(2,0,1)
    # normalization according to https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/applications/imagenet_utils.py#L259
    img = (img/127.5).astype(np.float32)
    img -= 1.
    image_tensor.byte_data = img.tobytes()
    image_tensor_metadata = list_model_tensors(list_models())[model_name]['inputs'][0]
    image_tensor.tensor_metadata.name = image_tensor_metadata.name
    image_tensor.tensor_metadata.data_type = image_tensor_metadata.data_type
    for shape in image_tensor_metadata.shape:
        image_tensor.tensor_metadata.shape.append(shape)
    predict_request = agent_pb2.PredictRequest()
    predict_request.name = model_name
    predict_request.tensors.append(image_tensor)
    predict_response = agent_client.Predict(predict_request)
    return predict_response

def main():
    try:
        unload_model(ml_model)
    except:
        pass
  
    print('LoadModel...', end='')
    try:
        load_model(ml_model, model_path=model_path)
        print('done.')
    except Exception as e:
        print()
        print(e)
        print('Model already loaded!')
        
    print('ListModels...', end='')
    try:
        print(list_models())
        print('done.')
        
    except Exception as e:
        print()
        print(e)
        print('List model failed!')

    print('Predict')
    try: 
        prediction = predict_image(ml_model, image_path=image_path)
        #print(prediction) # uncomment to print the predictio object
        pred = np.frombuffer(prediction.tensors[0].byte_data, dtype = np.float32)
        # the returned array has shape (1000,), while mobilenet v2 returns a shape (1, 1000)
        # decoding results https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/applications/imagenet_utils.py#L159
        
        top_indexes = pred.argsort()[-5:][::-1]
        with open(os.environ["IMAGENET_CLASS_INDEX_PATH"]) as f:
            classes = json.load(f)
        result = [tuple(classes[str(i)]) + (pred[i],) for i in top_indexes]
        print(result)
    except Exception as e:
        print()
        print(e)
        print('Predict failed!')
       
    print('Unload model...', end='')
    try:
        unload_model(ml_model)
        print('done.')
    except Exception as e:
        print()
        print(e)
        print('unload model failed!')

if __name__ == '__main__':
    main()
