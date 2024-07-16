# Import necessary libraries
import sys
import subprocess
import os
import warnings
import time
import argparse
import boto3
import pandas as pd
import sagemaker
import json
import glob
import os
import tensorflow as tf
import logging
import pathlib
from sagemaker.s3 import S3Downloader
from sagemaker.session import Session
from sklearn.model_selection import train_test_split
from defusedxml.ElementTree import parse

start_time = time.time()  # Record the start time for timing purposes

from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# This set will store unique class names from the dataset
class_ids = set()


# Function to parse an XML file and return a dictionary with image and annotation details
def xml_data_parser(xml_file):
    # Check if the XML file exists
    if os.path.isfile(xml_file):
        with open(xml_file) as f:
            # tree = ET.parse(f)
            tree = parse(f)
            root = tree.getroot()

        annotation_dict = {}
        filename = root.findall("filename")[0].text
        width = root.findall("size")[0].find("width").text
        height = root.findall("size")[0].find("height").text
        depth = root.findall("size")[0].find("depth").text
        annotation_dict["image"] = filename
        annotation_dict["width"] = width
        annotation_dict["height"] = height
        annotation_dict["annotation"] = []
        for i in range(len(root.findall("object"))):
            annotation = {}
            xmin = root.findall("object")[i].find("bndbox").find("xmin").text
            ymin = root.findall("object")[i].find("bndbox").find("ymin").text
            xmax = root.findall("object")[i].find("bndbox").find("xmax").text
            ymax = root.findall("object")[i].find("bndbox").find("ymax").text
            name = root.findall("object")[i].find("name").text
            annotation["category"] = name
            annotation["bbox"] = [xmin, ymin, xmax, ymax]
            annotation_dict["annotation"].append(annotation)
            class_ids.add(name)  # Add the class name to the set
        return annotation_dict
    else:
        print(f"{xml_file} doesnt exists")
        return {}


# Function to parse the dataset files and return a list of dictionaries
def parse_dataset_files(base_path, filename):
    # Load the file and create a list of filenames
    with open(filename, "r") as f:
        dataset_lines = f.readlines()
        dataset_files = [i.strip() for i in dataset_lines]

    # Create a list of XML file paths
    xml_files = [
        f"{base_path}/VOCdevkit/VOC2012/Annotations/{dataset_files[i]}.xml"
        for i in range(len(dataset_files))
    ]
    result = [xml_data_parser(xml_file) for xml_file in xml_files]
    return result


# Function to dump and load a dataset from/to a JSONL file
def dump_and_load_dataset(dataset, filename):
    # Dumps the list of dicts to a JSONL file
    with open(filename, "w") as outfile:
        json.dump(dataset, outfile)
    # Loads the JSONL file to a list of dicts structure
    with open(filename, "r") as f:
        output_dataset = json.load(f)
    return output_dataset


# Function to map class labels to integers
def class_text_to_int(label):
    if label in class_mapping_by_label.keys():
        return class_mapping_by_label[label]


# TFRecords helper functions
def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# Function to create a TFRecord example from a sample
def create_example(image, image_path, group):
    width = int(group["width"])
    height = int(group["height"])
    classes_text = []
    classes = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []

    for index, row in enumerate(group["annotation"]):
        xmin.append(float(row["bbox"][0]))
        ymin.append(float(row["bbox"][1]))
        xmax.append(float(row["bbox"][2]))
        ymax.append(float(row["bbox"][3]))
        classes.append(class_text_to_int(row["category"]))
        classes_text.append(row["category"].encode("utf8"))

    feature = tf.train.Example(
        features=tf.train.Features(
            feature={
                "height": int64_feature(height),
                "width": int64_feature(width),
                "filename": bytes_feature(group["image"].encode("utf8")),
                "image": image_feature(image),
                "object/bbox/xmin": float_list_feature(xmin),
                "object/bbox/xmax": float_list_feature(xmax),
                "object/bbox/ymin": float_list_feature(ymin),
                "object/bbox/ymax": float_list_feature(ymax),
                "object/text": bytes_list_feature(classes_text),
                "object/label": int64_list_feature(classes),
            }
        )
    )

    return feature


# Helper function to parse TFRecordDataset back to tensors
def parse_tfrecord_fn(example):
    feature_description = {
        "height": tf.io.FixedLenFeature((), tf.int64),
        "width": tf.io.FixedLenFeature((), tf.int64),
        "filename": tf.io.FixedLenFeature((), tf.string),
        "image": tf.io.FixedLenFeature((), tf.string),
        "object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "object/text": tf.io.VarLenFeature(tf.string),
        "object/label": tf.io.VarLenFeature(tf.int64),
    }
    # Parse the single example
    example = tf.io.parse_single_example(example, feature_description)
    # Preprocess the image and bounding box data
    example["image"] = tf.cast(tf.io.decode_jpeg(example["image"], channels=3), tf.float32)
    example["filename"] = tf.cast(example["filename"], tf.string)
    example["object/bbox/xmin"] = tf.sparse.to_dense(example["object/bbox/xmin"])
    example["object/bbox/xmax"] = tf.sparse.to_dense(example["object/bbox/xmax"])
    example["object/bbox/ymin"] = tf.sparse.to_dense(example["object/bbox/ymin"])
    example["object/bbox/ymax"] = tf.sparse.to_dense(example["object/bbox/ymax"])
    example["object/text"] = tf.sparse.to_dense(example["object/text"])
    example["object/label"] = tf.sparse.to_dense(example["object/label"])
    # Combine the bounding box coordinates into a single tensor
    example["object/bbox"] = tf.stack(
        [
            example["object/bbox/xmin"],
            example["object/bbox/ymin"],
            example["object/bbox/xmax"],
            example["object/bbox/ymax"],
        ],
        axis=1,
    )

    return example


# Helper function to prepare sample in the expected format
def prepare_sample(inputs):
    image = inputs["image"]
    boxes = inputs["object/bbox"]
    bounding_boxes = {
        "classes": inputs["object/label"],
        "boxes": boxes,
    }
    return {"images": image, "bounding_boxes": bounding_boxes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    print("Received arguments {}".format(args))

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/input").mkdir(parents=True, exist_ok=True)
    output_dir = base_dir + "/output"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("downloading file")

    input_path = f"{base_dir}/input/"

    # untar the partitioned data files into the data folder
    logger.info("extracting files")
    #os.system(f"tar -xf {input_path}/VOCtrainval_11-May-2012.tar -C {input_path}")
    subprocess.run(['tar','-xf',f'{input_path}/VOCtrainval_11-May-2012.tar','-C',input_path])

    # Creation of datasets (list of dicts)
    logger.info("Creating Json files")
    train_dataset = parse_dataset_files(
        input_path, f"{input_path}/VOCdevkit/VOC2012/ImageSets/Main/train.txt"
    )
    val_dataset = parse_dataset_files(
        input_path, f"{input_path}/VOCdevkit/VOC2012/ImageSets/Main/val.txt"
    )
    print(
        f"{len(train_dataset)} samples for training and {len(val_dataset)} samples for validation before loading"
    )

    # Dump of dicts to jsonl files and load for testing purposes
    train_filename = f"{output_dir}/train_labels_VOC.jsonl"
    val_filename = f"{output_dir}/val_labels_VOC.jsonl"
    train_dataset = dump_and_load_dataset(train_dataset, train_filename)
    # val_dataset=dump_and_load_dataset(val_dataset,val_filename)

    # Split the train dataset into train and validation sets
    training_dataset, val_dataset = train_test_split(train_dataset, test_size=0.33, random_state=42)

    print(
        f"{len(train_dataset)} samples for training and {len(val_dataset)} samples for validation after loading"
    )

    # creation of a dict to map classes with class ids
    class_ids = sorted(list(class_ids))
    class_mapping = dict(zip(range(len(class_ids)), class_ids))
    class_mapping_by_label = dict(zip(class_ids, range(len(class_ids))))

    logger.info("Creating TFRecords")
    # creates output folders
    tfrecords_dir = f"{output_dir}/tfrecords"
    if not os.path.exists(tfrecords_dir + "/train"):
        os.makedirs(tfrecords_dir + "/train")
    if not os.path.exists(tfrecords_dir + "/val"):
        os.makedirs(tfrecords_dir + "/val")

    num_samples = 1024  # number of samples on each TFRecord file
    num_tfrecords_train = len(train_dataset) // num_samples
    num_tfrecords_val = len(val_dataset) // num_samples
    if len(train_dataset) % num_samples:
        num_tfrecords_train += 1  # add one record if there are any remaining samples
    if len(val_dataset) % num_samples:
        num_tfrecords_val += 1  # add one record if there are any remaining samples

    images_dir = f"{input_path}/VOCdevkit/VOC2012/JPEGImages"
    logger.info("Creating Training Records")
    for tfrec_num in range(num_tfrecords_train):
        samples = train_dataset[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)]

        with tf.io.TFRecordWriter(
            tfrecords_dir + "/train/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
        ) as writer:
            for sample in samples:
                image_path = f"{images_dir}/{sample['image']}"
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                example = create_example(image, image_path, sample)
                writer.write(example.SerializeToString())
    logger.info("Creating validation records")
    for tfrec_num in range(num_tfrecords_val):
        samples = val_dataset[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)]

        with tf.io.TFRecordWriter(
            tfrecords_dir + "/val/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
        ) as writer:
            for sample in samples:
                image_path = f"{images_dir}/{sample['image']}"
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                example = create_example(image, image_path, sample)
                writer.write(example.SerializeToString())
    logger.info(f"Processed data save on {output_dir}")
