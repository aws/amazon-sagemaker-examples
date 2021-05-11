"""
Glue Script to tranfor worker metrics from Cloudwatch into Glue Catalog
"""
import sys

import pyspark.sql.functions as f
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import col, dayofmonth, from_json, hour, month, year
from pyspark.sql.types import StringType, StructField, StructType, TimestampType

sc = SparkContext()
glueContext = GlueContext(SparkContext.getOrCreate())
spark = glueContext.spark_session

job = Job(glueContext)
args = getResolvedOptions(sys.argv, ["JOB_NAME", "source_bucket_path", "destination_bucket"])
job.init(args["JOB_NAME"], args)

source_bucket_path = args["source_bucket_path"]
destination_bucket = args["destination_bucket"]

datasource0 = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": [source_bucket_path], "compressionType": "gzip", "recurse": True},
    # compressionType="gzip",
    format="csv",
    format_options={"separator": " "},
)

input_df = datasource0.toDF()

schema = StructType(
    [
        StructField("worker_id", StringType(), True),
        StructField("cognito_user_pool_id", StringType(), True),
        StructField("cognito_sub_id", StringType(), True),
        StructField("task_accepted_time", TimestampType(), True),
        StructField("task_submitted_time", StringType(), True),
        StructField("task_returned_time", StringType(), True),
        StructField("workteam_arn", StringType(), True),
        StructField("labeling_job_arn", StringType(), True),
        StructField("work_requester_account_id", StringType(), True),
        StructField("job_reference_code", StringType(), True),
        StructField("job_type", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("event_timestamp", StringType(), True),
    ]
)

if len(input_df.head(1)) > 0:
    df2 = (
        input_df.withColumnRenamed("col0", "timestamp")
        .withColumn("col1", from_json("col1", schema))
        .select(col("timestamp"), col("col1.*"))
    )

    partition_column = "task_accepted_time"
    df3 = (
        df2.withColumn("year", year(col(partition_column)))
        .withColumn("month", month(col("task_accepted_time")))
        .withColumn("date", dayofmonth(col("task_accepted_time")))
        .withColumn("hour", hour(col("task_accepted_time")))
    )

    print(df3.show(1))

    df3.write.mode("append").partitionBy("year", "month", "date", "hour").parquet(
        destination_bucket
    )

job.commit()
