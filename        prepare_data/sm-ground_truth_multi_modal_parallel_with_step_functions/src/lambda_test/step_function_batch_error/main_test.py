import os
import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

from step_functions_batch_error.main import lambda_handler
from test_shared.mock_objects import TestContext


class TestCase(TestCase):
    @patch("shared.db.mark_batch_and_children_failed")
    @patch.dict(os.environ, {"DEFAULT_STATUS_SNS_ARN": "TEST_ARN"})
    @patch("boto3.resource")
    def test_lambda_handler_happy_case(self, resource_mock, mark_batch_and_children_failed_mock):
        # Setup
        event = {
            "execution_id": "test",
            "input": {
                "error-info": {
                    "Cause": '{"errorType":"errorType", "errorMessage" : "errorMessage"}'
                },
                "transformation_step_output": {
                    "batch_id": "test",
                },
            },
        }

        mark_batch_and_children_failed_mock.return_value = {}
        mock_topic = Mock()
        mock_topic.publish.return_value = {}
        resource_mock.Topic.return_value = mock_topic
        # Act
        val = lambda_handler(event, TestContext())

        expected_output = {
            "output_sns_arn": "TEST_ARN",
            "published_sns": {
                "batchId": "test",
                "errorString": "errorMessage",
                "errorType": "errorType",
                "message": "Batch processing failed",
                "status": "FAILED",
                "token": "test",
            },
        }

        # Assert
        self.assertEqual(expected_output, val, "Unexpected response returned")


if __name__ == "__main__":
    unittest.main()
