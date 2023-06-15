import argparse
import csv
import os
import shutil
import sys
import time

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
    VectorIndexer,
)
from pyspark.sql.functions import *
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)
from pyspark.ml.functions import vector_to_array


def main():
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_input_bucket", type=str, help="s3 input bucket")
    parser.add_argument("--s3_input_key_prefix", type=str, help="s3 input key prefix")
    parser.add_argument("--s3_output_bucket", type=str, help="s3 output bucket")
    parser.add_argument("--s3_output_key_prefix", type=str, help="s3 output key prefix")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()

    # Defining the schema corresponding to the input data. The input data does not contain headers
    schema = StructType(
        [
            StructField("sex", StringType(), True),
            StructField("length", DoubleType(), True),
            StructField("diameter", DoubleType(), True),
            StructField("height", DoubleType(), True),
            StructField("whole_weight", DoubleType(), True),
            StructField("shucked_weight", DoubleType(), True),
            StructField("viscera_weight", DoubleType(), True),
            StructField("shell_weight", DoubleType(), True),
            StructField("rings", DoubleType(), True),
        ]
    )

    # Downloading the data from S3 into a Dataframe
    total_df = spark.read.csv(
        ("s3://" + os.path.join(args.s3_input_bucket, args.s3_input_key_prefix, "abalone.csv")),
        header=False,
        schema=schema,
    )

    # StringIndexer on the sex column which has categorical value
    sex_indexer = StringIndexer(inputCol="sex", outputCol="indexed_sex")

    # one-hot-encoding is being performed on the string-indexed sex column (indexed_sex)
    sex_encoder = OneHotEncoder(inputCol="indexed_sex", outputCol="sex_vec")

    # vector-assembler will bring all the features to a 1D vector for us to save easily into CSV format
    assembler = VectorAssembler(
        inputCols=[
            "sex_vec",
            "length",
            "diameter",
            "height",
            "whole_weight",
            "shucked_weight",
            "viscera_weight",
            "shell_weight",
        ],
        outputCol="features",
    )

    # add the above steps to a pipeline
    pipeline = Pipeline(stages=[sex_indexer, sex_encoder, assembler])

    # train the feature transformers
    model = pipeline.fit(total_df)

    # transform the dataset with information obtained from the previous fit
    transformed_total_df = model.transform(total_df)

    # split the overall dataset into 80-20 training and validation
    (train_df, validation_df) = transformed_total_df.randomSplit([0.8, 0.2])

    # extract only rings and features columns to write to csv
    train_df_final = train_df.withColumn("feature", vector_to_array("features")).select(
        ["rings"] + [col("feature")[i] for i in range(9)]
    )

    val_df_final = validation_df.withColumn("feature", vector_to_array("features")).select(
        ["rings"] + [col("feature")[i] for i in range(9)]
    )

    # write to csv files in S3
    train_df_final.write.csv(f"s3://{args.s3_output_bucket}/{args.s3_output_key_prefix}/train")
    val_df_final.write.csv(f"s3://{args.s3_output_bucket}/{args.s3_output_key_prefix}/validation")


if __name__ == "__main__":
    main()
