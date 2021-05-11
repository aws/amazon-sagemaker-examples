import unittest
from unittest import TestCase
from unittest.mock import patch

from labeling_job_state_change.main import lambda_handler
from test_shared.mock_objects import OutputTestData, TestContext


class TestCase(TestCase):
    @patch("shared.db.get_job_by_arn")
    @patch("shared.db.get_batch_metadata_by_labeling_job_name")
    @patch("shared.db.update_batch_status")
    @patch("shared.db.get_batch_metadata")
    def test_lambda_handler_happy_case(
        self,
        get_batch_metadata_mock,
        update_batch_status_mock,
        get_batch_metadata_by_labeling_job_name_mock,
        get_job_by_arn_mock,
    ):
        # Setup
        event = {"status": "test", "job_arns": ["arn:aws:sagemaker:test:test:labeling-job/test"]}
        get_batch_metadata_mock.return_value = OutputTestData.get_batch_first_level_output
        get_job_by_arn_mock.return_value = None
        get_batch_metadata_by_labeling_job_name_mock.return_value = (
            OutputTestData.get_batch_metadata_by_labeling_job_name_output
        )

        update_batch_status_mock.return_value = {}

        # Act
        val = lambda_handler(event, TestContext())

        # Assert
        self.assertEqual("success", val, "Unexpected status code returned")


if __name__ == "__main__":
    unittest.main()
