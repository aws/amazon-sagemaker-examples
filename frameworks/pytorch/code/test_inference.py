import json
import os
import shutil
import tarfile

import boto3
import botocore
import numpy as np
import sagemaker

from inference import input_fn, model_fn, output_fn, predict_fn


def fetch_model(model_data):
    """Untar the model.tar.gz object either from local file system
    or a S3 location

    Args:
        model_data (str): either a path to local file system starts with
        file:/// that points to the `model.tar.gz` file or an S3 link
        starts with s3:// that points to the `model.tar.gz` file

    Returns:
        model_dir (str): the directory that contains the uncompress model
        checkpoint files
    """

    model_dir = "/tmp/model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if model_data.startswith("file"):
        _check_model(model_data)
        shutil.copy2(
            os.path.join(model_dir, "model.tar.gz"), os.path.join(model_dir, "model.tar.gz")
        )
    elif model_data.startswith("s3"):
        # get bucket name and object key
        bucket_name = model_data.split("/")[2]
        key = "/".join(model_data.split("/")[3:])

        s3 = boto3.resource("s3")
        try:
            s3.Bucket(bucket_name).download_file(key, os.path.join(model_dir, "model.tar.gz"))
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print("the object does not exist.")
            else:
                raise

    # untar the model
    tar = tarfile.open(os.path.join(model_dir, "model.tar.gz"))
    tar.extractall(model_dir)
    tar.close()

    return model_dir


def test(model_data):
    # decompress the model.tar.gz file
    model_dir = fetch_model(model_data)

    # load the model
    net = model_fn(model_dir)

    # simulate some input data to test transform_fn

    data = {"inputs": np.random.rand(16, 1, 28, 28).tolist()}

    # encode numpy array to binary stream
    serializer = sagemaker.serializers.JSONSerializer()

    jstr = serializer.serialize(data)
    jstr = json.dumps(data)

    # "send" the bin_stream to the endpoint for inference
    # inference container calls transform_fn to make an inference
    # and get the response body for the caller

    content_type = "application/json"
    input_object = input_fn(jstr, content_type)
    predictions = predict_fn(input_object, net)
    res = output_fn(predictions, content_type)
    print(res)
    return


if __name__ == "__main__":
    model_data = "s3://sagemaker-us-west-2-688520471316/mxnet/mnist/pytorch-training-2020-11-21-22-02-56-203/model.tar.gz"
    test(model_data)
