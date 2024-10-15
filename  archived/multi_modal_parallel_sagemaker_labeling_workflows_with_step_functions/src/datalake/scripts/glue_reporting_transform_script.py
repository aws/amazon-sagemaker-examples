import sys

import boto3
import pyspark.sql.functions as f
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import (
    col,
    dayofmonth,
    from_json,
    hour,
    input_file_name,
    month,
    split,
    udf,
    unix_timestamp,
    year,
)
from pyspark.sql.types import ArrayType, StringType, StructField, StructType, TimestampType

sc = SparkContext()
glueContext = GlueContext(SparkContext.getOrCreate())
spark = glueContext.spark_session

job = Job(glueContext)
args = getResolvedOptions(
    sys.argv, ["JOB_NAME", "INPUT_PATH", "OUTPUT_PATH", "TRANSFORMATION_CTX", "POOL_ID"]
)

job.init(args["JOB_NAME"], args)


def username_sub_lookup(s):
    print(s)
    return username_sub_id_dict[s[0]]


def email_username_lookup(s):
    return username_email_dict[s]


input_path = args["INPUT_PATH"]
output_path = args["OUTPUT_PATH"]
transformation_ctx = args["TRANSFORMATION_CTX"]
poolId = args["POOL_ID"]

datasource0 = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": [input_path], "recurse": True},
    format="json",
    transformation_ctx=transformation_ctx,
)

region = poolId.split("_")[0]

cognito_client = boto3.client("cognito-idp", region)

cog_users = cognito_client.list_users(UserPoolId=poolId)
username_email_dict = {
    x["Username"]: y["Value"]
    for x in cog_users["Users"]
    for y in x["Attributes"]
    if y["Name"] == "email"
}
username_sub_id_dict = {
    y["Value"]: x["Username"]
    for x in cog_users["Users"]
    for y in x["Attributes"]
    if y["Name"] == "sub"
}

input_df = datasource0.toDF()

if len(input_df.head(1)) > 0:

    uname_udf = udf(username_sub_lookup)
    email_udf = udf(email_username_lookup)

    new_df = input_df.withColumn(
        "username", uname_udf(col("answers.workerMetadata.identityData.sub"))
    )
    new_df = new_df.withColumn("email", email_udf(new_df.username))
    new_df = new_df.withColumn("acceptanceTime", col("answers.acceptanceTime")[0])
    new_df = new_df.withColumn("submissionTime", col("answers.submissionTime")[0])
    new_df = new_df.withColumn("timeSpentInSeconds", col("answers.timeSpentInSeconds")[0])
    new_df = new_df.withColumn("workerId", col("answers.workerId")[0])
    new_df = new_df.withColumn(
        "identityProviderType", col("answers.workerMetadata.identityData.identityProviderType")[0]
    )
    new_df = new_df.withColumn("issuer", col("answers.workerMetadata.identityData.issuer")[0])
    new_df = new_df.withColumn("sub", col("answers.workerMetadata.identityData.sub")[0])
    new_df = new_df.withColumn("jobName", split(input_file_name(), "/").getItem(5))
    new_df = new_df.withColumn("modality", split(input_file_name(), "/").getItem(4))
    new_df = new_df.withColumn("answerContent", col("answers.answerContent"))
    new_df = new_df.drop("answers")

    df3 = (
        new_df.withColumn("year", year(col("submissionTime")))
        .withColumn("month", month(col("submissionTime")))
        .withColumn("date", dayofmonth(col("submissionTime")))
        .withColumn("hour", hour(col("submissionTime")))
    )

    # s3_output_path = 's3://{output_s3_bucket}/processed_worker_metrics/'
    df3.write.mode("append").partitionBy("year", "month", "date", "hour").parquet(output_path)
    # glueContext.write_dynamic_frame_from_options(frame=output_dynf,
    #                                              connection_type="s3",
    #                                              connection_options = {
    #                                                  "path": output_path,
    #                                                  "partitionKeys": partition_keys
    #                                              },
    #                                              format = "parquet")
else:
    print(f"The length of the data is empty")

job.commit()
