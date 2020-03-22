import os, glob
import sys

sys.path.append('/tensorpack/examples/FasterRCNN')

from modeling.generalized_rcnn import ResNetFPNModel
from config import finalize_configs, config as cfg
from eval import predict_image, DetectionResult
from dataset import register_coco

from tensorpack.predict.base import OfflinePredictor
from tensorpack.tfutils.sessinit import get_model_loader
from tensorpack.predict.config import PredictConfig

import numpy as np
import cv2
from itertools import groupby
from threading import Lock

class MaskRCNNService:

    lock = Lock()
    predictor = None

    # class method to load trained model and create an offline predictor
    @classmethod
    def get_predictor(cls):
        ''' load trained model'''

        with cls.lock:
            # check if model is already loaded
            if cls.predictor:
                return cls.predictor
        
            # create a mask r-cnn model
            mask_rcnn_model = ResNetFPNModel()

            try:
                model_dir = os.environ['SM_MODEL_DIR']
            except KeyError:
                model_dir = '/opt/ml/model'
            
            try:
                resnet_arch = os.environ['RESNET_ARCH']
            except KeyError:
                resnet_arch = 'resnet50'

            # file path to previoulsy trained mask r-cnn model
            latest_trained_model = ""
            model_search_path = os.path.join(model_dir, "model-*.index" )
            for model_file in glob.glob(model_search_path):
                if model_file > latest_trained_model:
                    latest_trained_model = model_file

            trained_model = latest_trained_model[:-6]
            print(f'Using model: {trained_model}')

            cfg.MODE_FPN = True
            cfg.MODE_MASK = True
            if resnet_arch == 'resnet101':
                cfg.BACKBONE.RESNET_NUM_BLOCKS = [3, 4, 23, 3]
            else:
                cfg.BACKBONE.RESNET_NUM_BLOCKS = [3, 4, 6, 3]
        
            cfg_prefix = "CONFIG__"
            for key,value in dict(os.environ).items():
                if key.startswith(cfg_prefix):
                    attr_name = key[len(cfg_prefix):]
                    attr_name = attr_name.replace('__', '.')
                    value=eval(value)
                    print(f"update config: {attr_name}={value}")
                    nested_var = cfg
                    attr_list = attr_name.split('.')
                    for attr in attr_list[0:-1]:
                        nested_var = getattr(nested_var, attr)
                    setattr(nested_var, attr_list[-1], value)
                    
            cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS
            cfg.DATA.BASEDIR = '/data'
            cfg.DATA.TRAIN='coco_train2017'
            cfg.DATA.VAL='coco_val2017'
            register_coco(cfg.DATA.BASEDIR)
            finalize_configs(is_training=False)

            # Create an inference model
            # PredictConfig takes a model, input tensors and output tensors
            input_tensors = mask_rcnn_model.get_inference_tensor_names()[0]
            output_tensors = mask_rcnn_model.get_inference_tensor_names()[1]
            
            cls.predictor = OfflinePredictor(PredictConfig(
                model=mask_rcnn_model,
                session_init=get_model_loader(trained_model),
                input_names=input_tensors,
                output_names=output_tensors))
            return cls.predictor

    # class method to predict
    @classmethod
    def predict(cls, img=None, img_id=None ):

        detection_results = predict_image(img, cls.predictor)
        print(detection_results)

        predictions={"img_id": str(img_id) }

        annotations=[]
        for dr in detection_results:
            a = {}
            b = dr.box.tolist()
            a["bbox"] = [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]
            category_id = dr.class_id
            
            a["category_id"] = int(category_id)
            a["category_name"] = cfg.DATA.CLASS_NAMES[int(category_id)]
            rle=cls.binary_mask_to_rle(dr.mask)
            a["segmentation"] = rle
            annotations.append(a)
        
        predictions["annotations"]=annotations
        print(predictions)

        return predictions

    @classmethod
    def binary_mask_to_rle(cls, binary_mask):
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='C'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
        return rle


# create predictor
MaskRCNNService.get_predictor()

import json
from flask import Flask
from flask import request
from flask import Response
import base64

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def health_check():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully and crrate a predictor."""
    health = MaskRCNNService.get_predictor() is not None  # You can insert a health check here

    status = 200 if health else 404
    return Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def inference():
    if not request.is_json:
        result = { "error": "Content type is not application/json"}
        print(result)
        return Response(response=result, status=415, mimetype='application/json')

    path = None
    try:
        content = request.get_json()
        img_id = content['img_id']
        path = os.path.join("/tmp", img_id)
        with open(path, "wb") as fh:
            ing_data_string = content["img_data"]
            img_data_bytes = bytearray(ing_data_string, encoding='utf-8')
            fh.write(base64.decodebytes(img_data_bytes))
            fh.close()
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            pred = MaskRCNNService.predict(img=img, img_id=img_id)

            return Response(response=json.dumps(pred), status=200, mimetype='application/json')
    except Exception as e:
        print(str(e))
        result = { "error": "Internal server error"}
        return Response(response=result, status=500, mimetype='application/json')
    finally:
        if path:
            os.remove(path)

