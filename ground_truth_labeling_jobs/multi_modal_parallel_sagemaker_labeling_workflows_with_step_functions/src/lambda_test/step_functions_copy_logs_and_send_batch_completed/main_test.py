import os
import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

from test_shared.mock_objects import OutputTestData, TestContext


class TestCase(TestCase):
    @patch("shared.api_helpers.input_batch_to_human_readable")
    @patch("shared.db.get_batch_metadata")
    @patch("shared.db.update_batch_status")
    @patch.dict(os.environ, {"DEFAULT_STATUS_SNS_ARN": "TEST_ARN"})
    @patch("boto3.resource")
    def test_lambda_handler_happy_case(
        self,
        resource_mock,
        update_batch_status_mock,
        get_batch_metadata_mock,
        input_batch_to_human_readable_mock,
    ):
        # Setup
        event = {
            "execution_id": "test",
            "batch_id": "test",
            "input": {
                "error-info": {
                    "Cause": '{"errorType":"errorType", "errorMessage" : "errorMessage"}'
                },
                "transformation_step_output": {
                    "batch_id": "test",
                },
            },
        }
        input_batch_to_human_readable_mock.return_value = (
            OutputTestData.input_batch_to_human_readable_output
        )
        get_batch_metadata_mock.return_value = OutputTestData.get_batch_metadata_output
        update_batch_status_mock.return_value = {}
        mock_topic = Mock()
        mock_topic.publish.return_value = {}
        resource_mock.Topic.return_value = mock_topic

        # Act
        from step_functions_copy_logs_and_send_batch_completed.main import lambda_handler

        val = lambda_handler(event, TestContext())

        # Assert
        expected_response = {
            "published_sns": {
                "batchId": "test",
                "message": "Batch processing has completed successfully.",
                "batchInfo": {
                    "batchId": "batch-test-non-streaming-13",
                    "status": "COMPLETE",
                    "inputLabelingJobs": [
                        {
                            "inputConfig": {"inputManifestS3Uri": "test"},
                            "jobLevel": 1,
                            "jobModality": "PointCloudObjectDetectionAudit",
                            "jobName": "batch-test-non-streaming-11-first",
                            "jobType": "BATCH",
                            "labelCategoryConfigS3Uri": "test",
                            "maxConcurrentTaskCount": 100,
                            "taskAvailabilityLifetimeInSeconds": 864000,
                            "taskTimeLimitInSeconds": 604800,
                            "workteamArn": "test",
                        },
                        {
                            "inputConfig": {
                                "chainFromJobName": "batch-test-non-streaming-11-first"
                            },
                            "jobLevel": 2,
                            "jobModality": "PointCloudObjectDetectionAudit",
                            "jobName": "batch-test-non-streaming-11-second",
                            "jobType": "BATCH",
                            "maxConcurrentTaskCount": 100,
                            "taskAvailabilityLifetimeInSeconds": 864000,
                            "taskTimeLimitInSeconds": 604800,
                            "workteamArn": "atest",
                        },
                        {
                            "inputConfig": {
                                "chainFromJobName": "batch-test-non-streaming-11-second"
                            },
                            "jobLevel": 3,
                            "jobModality": "PointCloudObjectDetectionAudit",
                            "jobName": "batch-test-non-streaming-11-third",
                            "jobType": "BATCH",
                            "maxConcurrentTaskCount": 100,
                            "taskAvailabilityLifetimeInSeconds": 864000,
                            "taskTimeLimitInSeconds": 604800,
                            "workteamArn": "test",
                        },
                    ],
                    "firstLevel": {
                        "status": "COMPLETE",
                        "numChildBatches": 1,
                        "numChildBatchesComplete": 1,
                        "jobLevels": [
                            {
                                "batchId": "12345",
                                "batchStatus": "COMPLETE",
                                "labelingJobName": "batch-test-non-streaming-11-second",
                                "labelAttributeName": "batch-test-non-streaming-11-second",
                                "labelCategoryS3Uri": "s3://testbucket/category-file.json",
                                "jobInputS3Uri": "s3://testbucket/category-file.json",
                                "jobInputS3Url": "https://testbucket.s3.amazonaws.com/category-file.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAY5FHYJAYKV7YHGUP%2F20210416%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210416T160942Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4045fba703131d01b8b7c2ad4fcfde9b34072e76200eb38b6eedcca6db5b6d72",
                                "jobOutputS3Uri": "s3://testbucket/category-file.json",
                                "jobOutputS3Url": "https://testbucket.s3.amazonaws.com/category-file.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAY5FHYJAYKV7YHGUP%2F20210416%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210416T160942Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4045fba703131d01b8b7c2ad4fcfde9b34072e76200eb38b6eedcca6db5b6d72",
                                "numFrames": 1,
                                "numFramesCompleted": 1,
                            }
                        ],
                    },
                    "secondLevel": {
                        "status": "COMPLETE",
                        "numChildBatches": 1,
                        "numChildBatchesComplete": 1,
                        "jobLevels": [
                            {
                                "batchId": "12345",
                                "batchStatus": "COMPLETE",
                                "labelingJobName": "batch-test-non-streaming-11-second",
                                "labelAttributeName": "batch-test-non-streaming-11-second",
                                "labelCategoryS3Uri": "s3://testbucket/category-file.json",
                                "jobInputS3Uri": "s3://testbucket/category-file.json",
                                "jobInputS3Url": "https://testbucket.s3.amazonaws.com/category-file.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAY5FHYJAYKV7YHGUP%2F20210416%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210416T160942Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4045fba703131d01b8b7c2ad4fcfde9b34072e76200eb38b6eedcca6db5b6d72",
                                "jobOutputS3Uri": "s3://testbucket/category-file.json",
                                "jobOutputS3Url": "https://testbucket.s3.amazonaws.com/category-file.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAY5FHYJAYKV7YHGUP%2F20210416%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210416T160942Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4045fba703131d01b8b7c2ad4fcfde9b34072e76200eb38b6eedcca6db5b6d72",
                                "numFrames": 1,
                                "numFramesCompleted": 1,
                            }
                        ],
                    },
                    "thirdLevel": {
                        "status": "COMPLETE",
                        "numChildBatches": 1,
                        "numChildBatchesComplete": 1,
                        "jobLevels": [
                            {
                                "batchId": "12345",
                                "batchStatus": "COMPLETE",
                                "labelingJobName": "batch-test-non-streaming-11-second",
                                "labelAttributeName": "batch-test-non-streaming-11-second",
                                "labelCategoryS3Uri": "s3://testbucket/category-file.json",
                                "jobInputS3Uri": "s3://testbucket/category-file.json",
                                "jobInputS3Url": "https://testbucket.s3.amazonaws.com/category-file.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAY5FHYJAYKV7YHGUP%2F20210416%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210416T160942Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4045fba703131d01b8b7c2ad4fcfde9b34072e76200eb38b6eedcca6db5b6d72",
                                "jobOutputS3Uri": "s3://testbucket/category-file.json",
                                "jobOutputS3Url": "https://testbucket.s3.amazonaws.com/category-file.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAY5FHYJAYKV7YHGUP%2F20210416%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210416T160942Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4045fba703131d01b8b7c2ad4fcfde9b34072e76200eb38b6eedcca6db5b6d72",
                                "numFrames": 1,
                                "numFramesCompleted": 1,
                            }
                        ],
                    },
                },
                "token": "test",
                "status": "SUCCESS",
            },
            "output_sns_arn": "TEST_ARN",
        }
        self.assertEqual(expected_response, val, "Unexpected response returned")


if __name__ == "__main__":
    unittest.main()
