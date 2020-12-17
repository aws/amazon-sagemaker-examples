from __future__ import print_function
from __future__ import unicode_literals

import time
import sys
import os
import shutil
import csv
import boto3

from awsglue.utils import getResolvedOptions

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StructField, StructType, StringType
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import *
from mleap.pyspark.spark_support import SimpleSparkSerializer

from awsglue.utils import getResolvedOptions

def csv_line(data):
    r = ' '.join(d for d in data[1])
    return ('__label__' + str(data[0])) + " " + r

def main():
    spark = SparkSession.builder.appName("DBPediaSpark").getOrCreate()

    args = getResolvedOptions(sys.argv, ['S3_INPUT_BUCKET',
                                         'S3_INPUT_KEY_PREFIX',
                                         'S3_OUTPUT_BUCKET',
                                         'S3_OUTPUT_KEY_PREFIX',
                                         'S3_MODEL_BUCKET',
                                         'S3_MODEL_KEY_PREFIX'])

    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    spark.sparkContext._jsc.hadoopConfiguration().set("mapred.output.committer.class",
                                                      "org.apache.hadoop.mapred.FileOutputCommitter")
    
    # Defining the schema corresponding to the input data. The input data does not contain the headers
    schema = StructType([StructField("label", IntegerType(), True), 
                         StructField("title", StringType(), True), 
                         StructField("abstract", StringType(), True)])
    
    # Download the data from S3 into two separate Dataframes
    traindf = spark.read.csv(('s3://' + os.path.join(args['S3_INPUT_BUCKET'], args['S3_INPUT_KEY_PREFIX'],
                                                   'train.csv')), header=False, schema=schema, encoding='UTF-8')
    validationdf = spark.read.csv(('s3://' + os.path.join(args['S3_INPUT_BUCKET'], args['S3_INPUT_KEY_PREFIX'],
                                                          'test.csv')), header=False, schema=schema, encoding='UTF-8')

    # Tokenize the abstract column which contains the input text
    tokenizer = Tokenizer(inputCol="abstract", outputCol="tokenized_abstract")

    # Save transformed training data to CSV in S3 by converting to RDD.
    transformed_traindf = tokenizer.transform(traindf)
    transformed_train_rdd = transformed_traindf.rdd.map(lambda x: (x.label, x.tokenized_abstract))
    lines = transformed_train_rdd.map(csv_line)
    lines.coalesce(1).saveAsTextFile('s3://' + os.path.join(args['S3_OUTPUT_BUCKET'], args['S3_OUTPUT_KEY_PREFIX'], 'train'))

    # Similar data processing for validation dataset.
    transformed_validation = tokenizer.transform(validationdf)
    transformed_validation_rdd = transformed_validation.rdd.map(lambda x: (x.label, x.tokenized_abstract))
    lines = transformed_validation_rdd.map(csv_line)
    lines.coalesce(1).saveAsTextFile('s3://' + os.path.join(args['S3_OUTPUT_BUCKET'], args['S3_OUTPUT_KEY_PREFIX'], 'validation'))

    # Serialize the tokenizer via MLeap and upload to S3
    SimpleSparkSerializer().serializeToBundle(tokenizer, "jar:file:/tmp/model.zip", transformed_validation)

    # Unzip as SageMaker expects a .tar.gz file but MLeap produces a .zip file.
    import zipfile
    with zipfile.ZipFile("/tmp/model.zip") as zf:
        zf.extractall("/tmp/model")

    # Write back the content as a .tar.gz file
    import tarfile
    with tarfile.open("/tmp/model.tar.gz", "w:gz") as tar:
        tar.add("/tmp/model/bundle.json", arcname='bundle.json')
        tar.add("/tmp/model/root", arcname='root')

    s3 = boto3.resource('s3')
    file_name = os.path.join(args['S3_MODEL_KEY_PREFIX'], 'model.tar.gz')
    s3.Bucket(args['S3_MODEL_BUCKET']).upload_file('/tmp/model.tar.gz', file_name)


if __name__ == "__main__":
    main()
