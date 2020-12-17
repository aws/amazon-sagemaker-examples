class AlgorithmArnProvider:
    
    @staticmethod
    def get_algorithm_arn(current_region):
        mapping = {            
          "ap-northeast-1" : "arn:aws:sagemaker:ap-northeast-1:977537786026:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "ap-northeast-2" : "arn:aws:sagemaker:ap-northeast-2:745090734665:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "ap-southeast-1" : "arn:aws:sagemaker:ap-southeast-1:192199979996:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "ap-southeast-2" : "arn:aws:sagemaker:ap-southeast-2:666831318237:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "us-east-1" : "arn:aws:sagemaker:us-east-1:865070037744:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "eu-central-1" : "arn:aws:sagemaker:eu-central-1:446921602837:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "ap-south-1" : "arn:aws:sagemaker:ap-south-1:077584701553:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "sa-east-1" : "arn:aws:sagemaker:sa-east-1:270155090741:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "ca-central-1" : "arn:aws:sagemaker:ca-central-1:470592106596:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "eu-west-1" : "arn:aws:sagemaker:eu-west-1:985815980388:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "eu-west-2" : "arn:aws:sagemaker:eu-west-2:856760150666:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "eu-west-3" : "arn:aws:sagemaker:eu-west-3:843114510376:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "eu-north-1" : "arn:aws:sagemaker:eu-north-1:136758871317:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "us-west-1" : "arn:aws:sagemaker:us-west-1:382657785993:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "us-east-2" : "arn:aws:sagemaker:us-east-2:057799348421:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9",
          "us-west-2" : "arn:aws:sagemaker:us-west-2:594846645681:algorithm/autogluon-tabular-v3-3-3b713280f53340e66804815d6707dbc9"
        }
        return mapping[current_region]