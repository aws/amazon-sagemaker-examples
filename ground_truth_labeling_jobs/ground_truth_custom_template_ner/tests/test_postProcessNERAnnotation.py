import json
from io import BytesIO
from unittest import TestCase
from unittest.mock import Mock

from postprocess_handler import PostProcessNERAnnotation


class TestPostProcessNERAnnotation(TestCase):

    def test_post_process(self):
        # Arrange
        sut = PostProcessNERAnnotation()

        event = {"version": "2018-10-06",
                 "labelingJobArn": "arn:aws:sagemaker:us-east-2:111:labeling-job/entityrecognition-protienname",
                 "payload": {
                     "s3Uri": "s3://mybucket/output/EntityRecognition-ProtienName/annotations/consolidated-annotation/consolidation-request/iteration-1/2018-12-20_11:42:13.json"},
                 "labelAttributeName": "EntityRecognition-ProtienName",
                 "roleArn": "arn:aws:iam::555:role/Sagemaker",
                 "outputConfig": "s3://mybucket/output/EntityRecognition-ProtienName/annotations"}

        payload = [

            {
                "datasetObjectId": "1",
                "dataObject": {
                    "content": "Protein tyrosine regulates insulin receptor protein."
                },
                "annotations": [
                    {
                        "workerId": "private.us-east-2.LMI3IGP47RB6RXWSMWOAYMG57U",
                        "annotationData": {
                            "content": '{"entities": "{\\"0\\":{\\"startindex\\":\\"10\\",\\"tokentext\\":\\"tyrosine\\"},\\"1\\":{\\"startindex\\":\\"15\\",\\"tokentext\\":\\"receptor\\"}}"}'

                        }
                    }
                ]
            },
            {
                "datasetObjectId": "2",
                "dataObject": {
                    "content": "Proteins klk3 regulates insulin receptor proteins."
                },
                "annotations": [
                    {
                        "workerId": "private.us-east-2.LMI3IGP47RB6RXWSMWOAYMG57U",
                        "annotationData": {
                            "content": '{"entities": "{\\"0\\":{\\"startindex\\":\\"10\\",\\"tokentext\\":\\"klk3\\"},\\"1\\":{\\"startindex\\":\\"15\\",\\"tokentext\\":\\"receptor\\"}}"}'

                        }
                    }
                ]
            }
        ]

        expected = [
            {
                "datasetObjectId": "1",
                "consolidatedAnnotation": {
                    "content": {
                        "EntityRecognition-ProtienName": {
                            "entities": [
                                {
                                    "start_index": "10",
                                    "length": 8,
                                    "token": "tyrosine"
                                },
                                {
                                    "start_index": "15",
                                    "length": 8,
                                    "token": "receptor"
                                }
                            ]
                        }
                    }
                }
            },
            {
                "datasetObjectId": "2",
                "consolidatedAnnotation": {
                    "content": {
                        "EntityRecognition-ProtienName": {
                            "entities": [
                                {
                                    "start_index": "10",
                                    "length": 4,
                                    "token": "klk3"
                                },
                                {
                                    "start_index": "15",
                                    "length": 8,
                                    "token": "receptor"
                                }
                            ]
                        }
                    }
                }
            }
        ]

        mock_s3_client = Mock()
        mock_s3_Object = Mock()
        mock_s3_Object.get.return_value = {"Body": BytesIO(json.dumps(payload).encode("utf-8"))}
        mock_s3_client.Object.return_value = mock_s3_Object
        sut.s3_client = mock_s3_client

        # Act
        actual = sut.process(event)

        # Assert
        self.assertEqual(actual, expected)
