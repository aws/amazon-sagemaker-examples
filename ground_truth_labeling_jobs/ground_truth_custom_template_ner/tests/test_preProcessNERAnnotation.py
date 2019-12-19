from unittest import TestCase

from preprocess_handler import PreProcessNERAnnotation


class TestPostProcessNERAnnotation(TestCase):

    def test_process(self):
        # Arrange
        sut = PreProcessNERAnnotation()

        event = {
            "version": "2018-10-06",
            "labelingJobArn": "arn:aws:sagemaker:us-east-2:1111:labeling-job/entityrecognition-clone-clone-clone-clone",
            "dataObject": {
                "source": "Protein tyrosine regulates insulin receptor protein."
            }
        }

        expected = [{'id': 0, 'startindex': 0, 'tokentext': 'Protein'},
                    {'id': 1, 'startindex': 8, 'tokentext': 'tyrosine'},
                    {'id': 2, 'startindex': 17, 'tokentext': 'regulates'},
                    {'id': 3, 'startindex': 27, 'tokentext': 'insulin'},
                    {'id': 4, 'startindex': 35, 'tokentext': 'receptor'},
                    {'id': 5, 'startindex': 44, 'tokentext': 'protein.'}]

        # Act
        actual = sut.process(event)

        # Assert
        self.assertEqual(actual, expected)
