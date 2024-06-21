class AlgorithmArnProvider:
    @staticmethod
    def get_algorithm_arn(current_region):
        algo_version = "autogluon-tabular-v3-5-cb7001bd0e8243b50adc3338deb44a48"
        region_mapping = {
            "ap-northeast-1": "arn:aws:sagemaker:ap-northeast-1:977537786026:algorithm/",
            "ap-northeast-2": "arn:aws:sagemaker:ap-northeast-2:745090734665:algorithm/",
            "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:algorithm/",
            "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:algorithm/",
            "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:algorithm/",
            "eu-central-1": "arn:aws:sagemaker:eu-central-1:446921602837:algorithm/",
            "ap-south-1": "arn:aws:sagemaker:ap-south-1:077584701553:algorithm/",
            "sa-east-1": "arn:aws:sagemaker:sa-east-1:270155090741:algorithm/",
            "ca-central-1": "arn:aws:sagemaker:ca-central-1:470592106596:algorithm/",
            "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:algorithm/",
            "eu-west-2": "arn:aws:sagemaker:eu-west-2:856760150666:algorithm/",
            "eu-west-3": "arn:aws:sagemaker:eu-west-3:843114510376:algorithm/",
            "eu-north-1": "arn:aws:sagemaker:eu-north-1:136758871317:algorithm/",
            "us-west-1": "arn:aws:sagemaker:us-west-1:382657785993:algorithm/",
            "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:algorithm/",
            "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:algorithm/",
        }
        return region_mapping[current_region] + algo_version
