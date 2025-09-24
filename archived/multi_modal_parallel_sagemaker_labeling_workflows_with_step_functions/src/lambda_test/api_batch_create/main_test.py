import json
import os
import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

from api_batch_create.main import lambda_handler
from botocore.exceptions import ClientError
from test_shared.mock_objects import InputTestData, TestContext


class TestCase(TestCase):
    def mock_sagemaker_api_call(self, operation_name, kwarg):
        if operation_name == "DescribeLabelingJob":
            parsed_response = {"Error": {"Code": "500", "Message": "Error Uploading"}}
            raise ClientError(parsed_response, operation_name)

    @patch("shared.db.get_batch_metadata")
    @patch("botocore.client.BaseClient._make_api_call", new=mock_sagemaker_api_call)
    @patch.dict(os.environ, {"DEFAULT_WORKTEAM_ARN": "TEST"})
    def test_lambda_handler_happyCase(self, get_batch_metadata_mock):
        # Setup
        event = Mock()
        event.get.return_value = json.dumps(InputTestData.create_batch_request)
        get_batch_metadata_mock.return_value = None

        # Act
        val = lambda_handler(event, TestContext())

        # Assert
        self.assertEqual(val["statusCode"], 200, "Unexpected status code returned")


if __name__ == "__main__":
    unittest.main()
