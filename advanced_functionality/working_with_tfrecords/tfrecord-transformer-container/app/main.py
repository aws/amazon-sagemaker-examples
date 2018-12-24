import codecs
import crcmod
import io
import json
import logging
import struct
import tensorflow as tf

from flask import Flask, Response, request
from google.protobuf.json_format import MessageToDict

app = Flask(__name__)


def _masked_crc32c(value):
    crc = crcmod.predefined.mkPredefinedCrcFun('crc-32c')(value)
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff


def read_tfrecords(tfrecords):
    tfrecords_bytes = io.BytesIO(tfrecords)

    examples = []

    while True:
      length_header = 12
      buf = tfrecords_bytes.read(length_header)
      if not buf:
        # reached end of tfrecord buffer, return examples
        return examples

      if len(buf) != length_header:
        raise ValueError('TFrecord is fewer than %d bytes' % length_header)
      length, length_mask = struct.unpack('<QI', buf)
      length_mask_actual = _masked_crc32c(buf[:8])
      if length_mask_actual != length_mask:
        raise ValueError('TFRecord does not contain a valid length mask')

      length_data = length + 4
      buf = tfrecords_bytes.read(length_data)
      if len(buf) != length_data:
        raise ValueError('TFRecord data payload has fewer bytes than specified in header')
      data, data_mask_expected = struct.unpack('<%dsI' % length, buf)
      data_mask_actual = _masked_crc32c(data)
      if data_mask_actual != data_mask_expected:
        raise ValueError('TFRecord has an invalid data crc32c')

      # Deserialize the tf.Example proto
      example = tf.train.Example()
      example.ParseFromString(data)

      # Extract a feature map from the example object
      example_feature = MessageToDict(example.features)['feature']
      feature_dict = {}
      for feature_key in example_feature.keys():
        feature_dict[feature_key] = example_feature[feature_key][list(example_feature[feature_key].keys())[0]]['value'][0]
      examples.append(feature_dict)


@app.route("/invocations", methods=['POST'])
def invocations():
    try:
      examples = read_tfrecords(request.data)
      # Build a TF serving predict request JSON
      response = Response(json.dumps({"signature_name": "predict", "instances": examples}))
      response.headers['Content-Type'] = "application/json"
      return response
    except ValueError as err:
      return str(err), 400


@app.route("/ping", methods=['GET'])
def ping():
    return "", 200
