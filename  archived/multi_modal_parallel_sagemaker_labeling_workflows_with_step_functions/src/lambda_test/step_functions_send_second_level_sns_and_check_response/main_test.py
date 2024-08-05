import os
import unittest
from unittest.mock import patch

from step_functions_send_second_level_sns_and_check_response.main import lambda_handler
from test_shared.mock_objects import OutputTestData, TestContext


class TestCase(unittest.TestCase):
    @patch.dict(os.environ, {"BATCH_PROCESSING_BUCKET_ID": "testBucket"})
    @patch("shared.db.insert_processed_input_batch_metadata")
    @patch("shared.db.get_batch_metadata")
    @patch("shared.db.get_batch_metadata_by_labeling_job_name")
    @patch("shared.s3_accessor.uri_to_s3_obj")
    @patch("shared.s3_accessor.fetch_s3")
    @patch("shared.s3_accessor.put_s3")
    def test_lambda_handler_happy_case(
        self,
        put_s3_mock,
        fetch_s3_mock,
        uri_to_s3_obj_mock,
        get_batch_metadata_by_labeling_job_name_mock,
        get_batch_metadata_mock,
        insert_processed_input_batch_metadata_mock,
    ):
        # Setup
        event = {
            "batch_id": "test",
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

        def side_effect(value):
            print(value)
            if value == "parent-batch":
                return OutputTestData.get_batch_metadata_output
            elif value == "test":
                return OutputTestData.get_batch_first_level_output
            else:
                raise Exception("unexpected argument provided")

        get_batch_metadata_mock.side_effect = side_effect
        get_batch_metadata_by_labeling_job_name_mock.return_value = (
            OutputTestData.get_batch_metadata_by_labeling_job_name_output
        )
        fetch_s3_mock.return_value = ("{}" + "\n" + "{}").encode()

        put_s3_mock.return_value = {}
        uri_to_s3_obj_mock.return_value = {}
        insert_processed_input_batch_metadata_mock.return_value = {}

        # Act
        from step_functions_send_second_level_sns_and_check_response.main import lambda_handler

        val = lambda_handler(event, TestContext())

        # Assert
        self.assertEqual(None, val, "Unexpected response returned")


if __name__ == "__main__":
    unittest.main()
