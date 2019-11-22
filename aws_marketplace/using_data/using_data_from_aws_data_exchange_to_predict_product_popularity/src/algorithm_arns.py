
class AlgorithmArnProvider:
    
    @staticmethod
    def get_decision_forest_algorithm_arn(current_region):
        mapping = {
            "ap-south-1":"arn:aws:sagemaker:ap-south-1:077584701553:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"ap-northeast-2":"arn:aws:sagemaker:ap-northeast-2:745090734665:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"ap-southeast-1":"arn:aws:sagemaker:ap-southeast-1:192199979996:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"ap-southeast-2":"arn:aws:sagemaker:ap-southeast-2:666831318237:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"ap-northeast-1":"arn:aws:sagemaker:ap-northeast-1:977537786026:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"ca-central-1":"arn:aws:sagemaker:ca-central-1:470592106596:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"eu-central-1":"arn:aws:sagemaker:eu-central-1:446921602837:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"eu-west-1":"arn:aws:sagemaker:eu-west-1:985815980388:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"eu-west-2":"arn:aws:sagemaker:eu-west-2:856760150666:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"us-east-1":"arn:aws:sagemaker:us-east-1:865070037744:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"us-east-2":"arn:aws:sagemaker:us-east-2:057799348421:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"us-west-1":"arn:aws:sagemaker:us-west-1:382657785993:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"us-west-2":"arn:aws:sagemaker:us-west-2:594846645681:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"sa-east-1":"arn:aws:sagemaker:sa-east-1:270155090741:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"eu-west-3":"arn:aws:sagemaker:eu-west-3:843114510376:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a",
"eu-north-1":"arn:aws:sagemaker:eu-north-1:136758871317:algorithm/intel-daal-decision-forest-cla-0b39950742dcac47e23d76a813c3d23a"
        }
        return mapping[current_region]