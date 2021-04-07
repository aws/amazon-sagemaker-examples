#!/usr/bin/env python

import boto3
    
import base64
from cryptography.fernet import Fernet
import os
import json
import pickle

import pprint

# init a boto3 session
sess = boto3.Session(
    aws_access_key_id=os.environ['ACCESS_KEY'],
    aws_secret_access_key=os.environ['SECRET_KEY'],
    region_name=os.environ['REGION_NAME']
)

# create s3 and kms client
s3 = sess.client('s3')
kms = sess.client('kms')


# a dummy ML model
class Model(object):
    def __init__(self):
        pass
        

# print out dictionary in a nice way
pp = pprint.PrettyPrinter(indent=1)


# where SageMaker injects training info inside container
input_dir="/opt/ml/input/"

# where SageMaker injects train / validation /test data
data_dir = '/opt/ml/input/data/'

# SageMaker treat "/opt/ml/model" as checkpoint direcotry
# and it will send everything there to S3 output path you 
# specified 
model_dir="/opt/ml/model"


def encrypt(data:bytes, plaintext_key:bytes) -> bytes:
    """Encrypt a chunk of bytes on client-side
    
    Args:
        data: a chunk of bytes
        plaintext_key: plaintext data key
        
    Return:
        encrypted data
    """
    ascii_str = base64.b64encode(plaintext_key)
    f = Fernet(key=ascii_str) 
    return f.encrypt(data)


def decrypt(data:bytes, plaintext_key:bytes) -> bytes:
    """Decrypt a chunk of bytes on client-side
    
    Args:
        data: encypted binary data
        plaintext_key: plaintext data key
    
    Return:
        decrypted data
    """    
    # to Fernet-friendly key
    ascii_str = base64.b64encode(plaintext_key)
    
    f = Fernet(key=ascii_str)
    return f.decrypt(data)

    
def get_data_key_ciphertext(bucket:str, key:str) -> bytes:
    """Get ciphertext data key from an S3 bucket
    
    Args:
        bucket: name of the S3 bucket
        key: object key in the S3 bucket (not to be confused with encryption key)
        
    Return: 
        data key ciphertext
    """
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj['Body'].read()

def get_data_key_plaintext(master_key:str, ciphertext: bytes) -> bytes:
    """Decrypt ciphertext data key
    
    Args:
        master_key: id of the master key
        ciphertext: ciphertext of the data key
    
    Return:
        plaintext data key
    """
    
    plaintext = kms.decrypt(
        KeyId=master_key, 
        CiphertextBlob=ciphertext)['Plaintext']
    return plaintext
    
    
def main():
    print("== Loading hyperparamters ===")
    with open(
        os.path.join(input_dir, 'config/hyperparameters.json'), 'rb') as f:
        hyp = json.load(f)
    
    
    # get ciphertext data key
    C = get_data_key_ciphertext(
        bucket=hyp['key_bucket'], 
        key=hyp['ciphertext_s3_key']
    )
    
    # get plaintext data key
    P = get_data_key_plaintext(
        master_key=hyp['master_key_id'],
        ciphertext=C
    )
    
    # read encrypted binary data
    data_file = os.path.join(
        data_dir, hyp['train_channel'], hyp['train_file'])
    
    with open(data_file, 'rb') as f:
        data = f.read()
        
    # if you do not decrypt the data 
    # then python cannot deserialize it to an object
    try:
        print("=== deserializing encrypted data (going to fail) ===")
        pickle.loads(data)
    except Exception as e:
        print('Failed to unpickle')
        print(e)
    
    # decrypt your training data
    plain_data = decrypt(data=data, plaintext_key=P)

    # bytes to python object
    py_data = pickle.loads(plain_data)
    
    print("=== decrypted data === ")
    print(py_data)
    
    
    # define your training logic here
    # import tensorflow as pd
    # import pandas as tf
    
    # suppose after the training is done
    # we get the following model 
    # don't think of it as sklearn model, pytorch model or whatever framework model
    # think of it as a serializable python object, i.e. a long sequence of 0 and 1 
    # that captures all the information about your model
    
    model = Model() 
    model_bytes = pickle.dumps(model)
    
    # encrypt 
    enc_model_bytes = encrypt(model_bytes, plaintext_key=P)
    
    print("== Saving model checkpoint ==")
    with open(os.path.join(model_dir, 'model.bin'), 'wb') as f:
        f.write(enc_model_bytes)
    
    print("== training completed ==")
    return

if __name__ == '__main__':
    main()


