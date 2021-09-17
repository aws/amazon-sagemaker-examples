import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

from api_batch_show.main import lambda_handler
from test_shared.mock_objects import InputTestData, OutputTestData, TestContext


class TestCase(TestCase):
    @patch("shared.db.get_batch_metadata")
    @patch("shared.db.get_child_batch_metadata")
    def test_lambda_handler_happy_case(
        self, get_child_batch_metadata_mock, get_batch_metadata_mock
    ):
        # Setup
        event = Mock()
        event.get.return_value = InputTestData.show_batch_request
        get_batch_metadata_mock.return_value = OutputTestData.get_batch_metadata_output
        get_child_batch_metadata_mock.return_value = OutputTestData.get_child_batch_metadata_output

        # Act
        val = lambda_handler(event, TestContext())

        # Assert
        self.assertEqual(200, val["statusCode"], "Unexpected status code returned")


if __name__ == "__main__":
    unittest.main()
