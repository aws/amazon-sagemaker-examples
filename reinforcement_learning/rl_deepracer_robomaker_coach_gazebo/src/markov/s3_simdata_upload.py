from __future__ import print_function

import logging
import os
import sys
from io import StringIO
import csv
import boto3

from markov import utils

logger = utils.Logger(__name__, logging.INFO).get_logger()

# Type of worker
SIMULATION_WORKER = "SIMULATION_WORKER"
SAGEMAKER_TRAINING_WORKER = "SAGEMAKER_TRAINING_WORKER"

#Defines to upload SIM_TRACE data in S3
SIMTRACE_CSV_DATA_HEADER = ['episode', 'steps', 'X', 'Y', 'yaw', 'steer', 'throttle', 'action', 'reward','done', 'all_wheels_on_track', 'progress', 'closest_waypoint', 'track_len', 'tstamp']
SIMTRACE_DATA_UPLOAD_UNKNOWN_STATE = 0
SIMTRACE_DATA_UPLOAD_INIT_DONE = 1
SIMTRACE_DATA_MPU_UPLOAD_IN_PROGRESS = 2
SIMTRACE_DATA_UPLOAD_DONE = 3

SIMTRACE_DATA_SIZE_ZERO = 0
SIMTRACE_DATA_MPU_MINSIZE = 5 * 1024 * 1024

node_type = os.environ.get("NODE_TYPE", SIMULATION_WORKER)
if node_type == SIMULATION_WORKER:
    import rospy

    class DeepRacerRacetrackSimTraceData():
        def __init__(self, s3_bucket, s3_key):
            logger.info("simtrace_data init")
            DeepRacerRacetrackSimTraceData.__instance = self
            self.data_state = SIMTRACE_DATA_UPLOAD_UNKNOWN_STATE
            self.s3_bucket = s3_bucket
            self.s3_object_key = s3_key
            if s3_key != "None":
                self.setup_mutipart_upload()

        def setup_mutipart_upload(self):
            logger.info("simtrace_data: setup_mutipart_upload to %s", self.s3_bucket)

            #setup for SIM_TRACE data incremental uploads to S3
            self.simtrace_csv_data = StringIO()
            self.csvwriter = csv.writer(self.simtrace_csv_data)
            self.csvwriter.writerow(SIMTRACE_CSV_DATA_HEADER)

            self.aws_region = rospy.get_param('AWS_REGION')
            logger.info("simtrace_data: setup_mutipart_upload on s3_bucket {} s3_object_key {} region {}".format(self.s3_bucket, self.s3_object_key, self.aws_region))

            #initiate the multipart upload
            s3_client = boto3.session.Session().client('s3', region_name=self.aws_region, config=utils.get_boto_config())
            self.mpu = s3_client.create_multipart_upload(Bucket=self.s3_bucket, Key=self.s3_object_key)
            self.mpu_id = self.mpu["UploadId"]
            self.mpu_part_number = 1
            self.mpu_parts = []
            self.mpu_episodes = 0
            self.total_upload_size = 0
            self.data_state = SIMTRACE_DATA_UPLOAD_INIT_DONE
            logger.info("simtrace_data: setup_mutipart_upload done! mpu_id= %s mpu_part_number", self.mpu_id)

        def write_simtrace_data(self,jsondata):
            if self.data_state != SIMTRACE_DATA_UPLOAD_UNKNOWN_STATE:
                try:
                    csvdata = []
                    for key in SIMTRACE_CSV_DATA_HEADER:
                        csvdata.append(jsondata[key])
                    self.csvwriter.writerow(csvdata)
                    self.total_upload_size += sys.getsizeof(csvdata)
                    logger.debug ("csvdata={} size data={} csv={}".format(csvdata, sys.getsizeof(csvdata), sys.getsizeof(self.simtrace_csv_data.getvalue())))
                except Exception:
                    utils.log_and_exit("Invalid SIM_TRACE data format",
                                       utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                                       utils.SIMAPP_EVENT_ERROR_CODE_500)

        def get_mpu_part_size(self):
            if self.data_state != SIMTRACE_DATA_UPLOAD_UNKNOWN_STATE:
                upload_size = sys.getsizeof(self.simtrace_csv_data.getvalue())
                logger.debug ("simtrace_data: upload size ={}".format(upload_size))
                return upload_size
            else:
                logger.info ("simtrace_data: invalid call to get_mpu_part_size")
                return SIMTRACE_DATA_SIZE_ZERO

        def reset_mpu_part(self, episodes):
            if self.data_state != SIMTRACE_DATA_UPLOAD_UNKNOWN_STATE:
                logger.debug("simtrace_data: reset_episode::: episode {}".format(episodes))
                self.simtrace_csv_data.close()
                self.simtrace_csv_data = StringIO()
                self.csvwriter = csv.writer(self.simtrace_csv_data)
                logger.info("simtrace_data: reset_part_upload::: done! episode {}".format(episodes))

        def upload_mpu_part_to_s3(self,episodes):
            if self.data_state != SIMTRACE_DATA_UPLOAD_UNKNOWN_STATE:
                logger.debug("simtrace_data: Uploading mpu_part_to_s3::: mpu_id-{} mpu_part_number-{} episode-{}".format(self.mpu_id, self.mpu_part_number,episodes))
                self.mpu_episodes = episodes
                s3_client = boto3.session.Session().client('s3', region_name=self.aws_region,
                                                           config=utils.get_boto_config())
                metrics_body = self.simtrace_csv_data.getvalue()
                part = s3_client.upload_part(
                       Body=bytes(metrics_body, encoding='utf-8'),
                       Bucket=self.s3_bucket,
                       Key=self.s3_object_key,
                       UploadId=self.mpu_id,
                       PartNumber=self.mpu_part_number)
                self.mpu_parts.append({"PartNumber": self.mpu_part_number, "ETag": part["ETag"]})
                self.mpu_part_number +=1
                logger.info("simtrace_data: Uploaded mpu_part_to_s3::: done! episode-{} mpu_id-{} mpu_part_number-{} mpu_parts-{}".format(episodes, self.mpu_id, self.mpu_part_number,self.mpu_parts))

        def upload_to_s3(self,episodes):
            if self.data_state != SIMTRACE_DATA_UPLOAD_UNKNOWN_STATE and self.data_state != SIMTRACE_DATA_UPLOAD_DONE:
                part_size = self.get_mpu_part_size()
                if part_size >= SIMTRACE_DATA_MPU_MINSIZE:
                    logger.info("simtrace_data: upload_to_s3::: episode-{} part_size-{} mpu_part_number-{}".format(episodes, part_size, self.mpu_part_number))
                    self.data_state = SIMTRACE_DATA_MPU_UPLOAD_IN_PROGRESS
                    self.upload_mpu_part_to_s3 (episodes)
                    self.reset_mpu_part(episodes)
                else:
                    logger.info("simtrace_data: upload_to_s3::: episode-{} part_size-{}, will upload after".format(episodes, part_size))

        def complete_upload_to_s3(self):
            logger.info("simtrace_data: complete_upload_to_s3::: data_state-{}".format(self.data_state))

            try:
                if self.data_state == SIMTRACE_DATA_MPU_UPLOAD_IN_PROGRESS:
                    #Multi-part upload to s3
                    self.data_state = SIMTRACE_DATA_UPLOAD_DONE
                    logger.info("simtrace_data: complete_upload_to_s3::Multi-part upload to S3 in progress, upload the last part number-{}, then complete mpu".format(self.mpu_part_number))
                    self.upload_mpu_part_to_s3 (self.mpu_episodes)

                    #now complete the multi-part-upload
                    session = boto3.session.Session()
                    s3_client = session.client('s3', region_name=self.aws_region,
                                               config=utils.get_boto_config())
                    result = s3_client.complete_multipart_upload(
                                Bucket=self.s3_bucket,
                                Key=self.s3_object_key,
                                UploadId=self.mpu_id,
                                MultipartUpload={"Parts": self.mpu_parts})
                    self.data_state = SIMTRACE_DATA_UPLOAD_DONE
                    logger.info("simtrace_data: complete_upload_to_s3 ::: multi-part-upload done,total raw size={}bytes result={}".format(self.total_upload_size, result))
                else:
                    #One-time upload to s3
                    if self.data_state == SIMTRACE_DATA_UPLOAD_INIT_DONE and self.data_state != SIMTRACE_DATA_UPLOAD_DONE:
                        self.data_state = SIMTRACE_DATA_UPLOAD_DONE
                        logger.info("simtrace_data:  complete_upload_to_s3 ::: write simtrace data to s3")
                        session = boto3.session.Session()
                        s3_client = session.client('s3', region_name=self.aws_region,
                                                   config=utils.get_boto_config())

                        # cancel multipart upload process
                        logger.info("simtrace_data: multi-part upload not required, cancel it before uploading the complete S3 object")
                        s3_client.abort_multipart_upload(
                                  Bucket=self.s3_bucket, Key=self.s3_object_key, UploadId=self.mpu_id)
                        metrics_body = self.simtrace_csv_data.getvalue()
                        logger.info("simtrace_data: complete_upload_to_s3:: write to s3 csv-formatted-data size={}bytes".format(sys.getsizeof(metrics_body)))
                        result = s3_client.put_object(
                                    Bucket=self.s3_bucket,
                                    Key=self.s3_object_key,
                                    Body=bytes(metrics_body, encoding='utf-8')
                                )
                        logger.info("simtrace_data: complete_upload_to_s3:: done writing simtrace total-unformatted-data size={}bytes to s3. result{}".format(self.total_upload_size, result))
                self.reset_mpu_part(self.mpu_episodes)
            except Exception as e:
                logger.info("simtrace_data: complete_upload_to_s3:: exception-{} ".format(e))
