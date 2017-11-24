import struct
import io
import boto3
import sys

import AmazonAIAlgorithmsIO_pb2
from record_pb2 import Record


def write_recordio(f, data):
    kmagic = 0xced7230a
    length = len(data)
    f.write(struct.pack('I', kmagic))
    f.write(struct.pack('I', length))
    upper_align = ((length + 3) >> 2) << 2
    padding = bytes([0x00 for _ in range(upper_align - length)])
    f.write(data)
    f.write(padding)


def list_to_record_bytes(values, keys=None, label=None, feature_size=None):
    record = Record()

    record.features['values'].float32_tensor.values.extend(values)

    if keys is not None:
        if feature_size is None:
            raise ValueError("For sparse tensors the feature size must be specified.")

        record.features['values'].float32_tensor.keys.extend(keys)

    if feature_size is not None:
        record.features['values'].float32_tensor.shape.extend([feature_size])

    if label is not None:
        record.label['values'].float32_tensor.values.extend([label])

    return record.SerializeToString()


def read_next(f):
     kmagic = 0xced7230a
     raw_bytes = f.read(4)
     if not raw_bytes:
         return
     m = struct.unpack('I', raw_bytes)[0]
     if m != kmagic:
         raise ValueError("Incorrect encoding")
     length = struct.unpack('I', f.read(4))[0]
     upper_align = ((length + 3) >> 2) << 2
     data = f.read(upper_align)
     return data[:length]

 
def to_proto(f, labels, vectors):
     for label, vec in zip(labels, vectors):
         record = AmazonAIAlgorithmsIO_pb2.Record()
         record.values.extend(vec)
         record.label = label
         write_recordio(f, record.SerializeToString())
 

def to_libsvm(f, labels, values):
     f.write('\n'.join(
         ['{} {}'.format(label, ' '.join(['{}:{}'.format(i + 1, el) for i, el in enumerate(vec)])) for label, vec in
          zip(labels, values)]))
     return f


def write_to_s3(fobj, bucket, key):
    return boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(fobj)


def upload_to_s3(partition_name, partition, bucket):
    labels = [t.tolist() for t in partition[1]]
    vectors = [t.tolist() for t in partition[0]]
    f = io.BytesIO()
    to_proto(f, labels, vectors)
    f.seek(0)
    key = "{}/examples".format(partition_name)
    url = 's3n://{}/{}'.format(bucket, key)
    print('Writing to {}'.format(url))
    write_to_s3(f, bucket, key)
    print('Done writing to {}'.format(url))


def convert_data(partitions, bucket):
    for partition_name, partition in partitions:
        print('{}: {} {}'.format(partition_name, partition[0].shape, partition[1].shape))
        upload_to_s3(partition_name, partition, bucket)
