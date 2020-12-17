from __future__ import print_function

import time
import sys
import os
import shutil
import csv
import boto3

from awsglue.utils import getResolvedOptions

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
from mleap.pyspark.spark_support import SimpleSparkSerializer


def toCSVLine(data):
    r = ','.join(str(d) for d in data[1])
    return str(data[0]) + "," + r


def main():
    spark = SparkSession.builder.appName("PySparkTitanic").getOrCreate()
    
    args = getResolvedOptions(sys.argv, ['s3_input_data_location',
                                         's3_output_bucket',
                                         's3_output_bucket_prefix', 
                                         's3_model_bucket',
                                         's3_model_bucket_prefix'])
    
    # This is needed to write RDDs to file which is the only way to write nested Dataframes into CSV.
    spark.sparkContext._jsc.hadoopConfiguration().set("mapred.output.committer.class",
                                                      "org.apache.hadoop.mapred.FileOutputCommitter")
    
    train = spark.read.csv(args['s3_input_data_location'], header=False)
    
    
    oldColumns = train.schema.names
    newColumns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'cat']

    train = reduce(lambda train, idx: train.withColumnRenamed(oldColumns[idx], newColumns[idx]), xrange(len(oldColumns)), train)
    
    # dropping null values
    train = train.dropna()
    
    # Target label
    catIndexer = StringIndexer(inputCol="cat", outputCol="label")
    
    labelIndexModel = catIndexer.fit(train)
    train = labelIndexModel.transform(train)
    
    converter = IndexToString(inputCol="label", outputCol="cat")

    # Spliting in train and test set. Beware : It sorts the dataset
    (traindf, validationdf) = train.randomSplit([0.8, 0.2])
    
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    buyingIndexer = StringIndexer(inputCol="buying", outputCol="indexedBuying")
    maintIndexer = StringIndexer(inputCol="maint", outputCol="indexedMaint")
    doorsIndexer = StringIndexer(inputCol="doors", outputCol="indexedDoors")
    personsIndexer = StringIndexer(inputCol="persons", outputCol="indexedPersons")
    lug_bootIndexer = StringIndexer(inputCol="lug_boot", outputCol="indexedLug_boot")
    safetyIndexer = StringIndexer(inputCol="safety", outputCol="indexedSafety")
    

    # One Hot Encoder on indexed features
    buyingEncoder = OneHotEncoder(inputCol="indexedBuying", outputCol="buyingVec")
    maintEncoder = OneHotEncoder(inputCol="indexedMaint", outputCol="maintVec")
    doorsEncoder = OneHotEncoder(inputCol="indexedDoors", outputCol="doorsVec")
    personsEncoder = OneHotEncoder(inputCol="indexedPersons", outputCol="personsVec")
    lug_bootEncoder = OneHotEncoder(inputCol="indexedLug_boot", outputCol="lug_bootVec")
    safetyEncoder = OneHotEncoder(inputCol="indexedSafety", outputCol="safetyVec")

    # Create the vector structured data (label,features(vector))
    assembler = VectorAssembler(inputCols=["buyingVec", "maintVec", "doorsVec", "personsVec", "lug_bootVec", "safetyVec"], outputCol="features")

    # Chain featurizers in a Pipeline
    pipeline = Pipeline(stages=[buyingIndexer, maintIndexer, doorsIndexer, personsIndexer, lug_bootIndexer, safetyIndexer, buyingEncoder, maintEncoder, doorsEncoder, personsEncoder, lug_bootEncoder, safetyEncoder, assembler])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(traindf)
    
    # Delete previous data from output
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(args['s3_output_bucket'])
    
    bucket.objects.filter(Prefix=args['s3_output_bucket_prefix']).delete()    

    # Save transformed training data to CSV in S3 by converting to RDD.
    transformed_traindf = model.transform(traindf)
    transformed_train_rdd = transformed_traindf.rdd.map(lambda x: (x.label, x.features))
    lines = transformed_train_rdd.map(toCSVLine)
    lines.saveAsTextFile('s3a://' + args['s3_output_bucket'] + '/' +args['s3_output_bucket_prefix'] + '/' + 'train')
    
    # Similar data processing for validation dataset.
    predictions = model.transform(validationdf)
    transformed_train_rdd = predictions.rdd.map(lambda x: (x.label, x.features))
    lines = transformed_train_rdd.map(toCSVLine)
    lines.saveAsTextFile('s3a://' + args['s3_output_bucket'] + '/' +args['s3_output_bucket_prefix'] + '/' + 'validation')

    # Serialize and store via MLeap  
    SimpleSparkSerializer().serializeToBundle(model, "jar:file:/tmp/model.zip", predictions)
    
    # Unzipping as SageMaker expects a .tar.gz file but MLeap produces a .zip file.
    import zipfile
    with zipfile.ZipFile("/tmp/model.zip") as zf:
        zf.extractall("/tmp/model")

    # Writing back the content as a .tar.gz file
    import tarfile
    with tarfile.open("/tmp/model.tar.gz", "w:gz") as tar:
        tar.add("/tmp/model/bundle.json", arcname='bundle.json')
        tar.add("/tmp/model/root", arcname='root')

    s3 = boto3.resource('s3')
    file_name = args['s3_model_bucket_prefix'] + '/' + 'model.tar.gz'
    s3.Bucket(args['s3_model_bucket']).upload_file('/tmp/model.tar.gz', file_name)

    os.remove('/tmp/model.zip')
    os.remove('/tmp/model.tar.gz')
    shutil.rmtree('/tmp/model')
    
    # Save postprocessor
    SimpleSparkSerializer().serializeToBundle(converter, "jar:file:/tmp/postprocess.zip", predictions)

    with zipfile.ZipFile("/tmp/postprocess.zip") as zf:
        zf.extractall("/tmp/postprocess")

    # Writing back the content as a .tar.gz file
    import tarfile
    with tarfile.open("/tmp/postprocess.tar.gz", "w:gz") as tar:
        tar.add("/tmp/postprocess/bundle.json", arcname='bundle.json')
        tar.add("/tmp/postprocess/root", arcname='root')

    file_name = args['s3_model_bucket_prefix'] + '/' + 'postprocess.tar.gz'
    s3.Bucket(args['s3_model_bucket']).upload_file('/tmp/postprocess.tar.gz', file_name)

    os.remove('/tmp/postprocess.zip')
    os.remove('/tmp/postprocess.tar.gz')
    shutil.rmtree('/tmp/postprocess')


if __name__ == "__main__":
    main()
