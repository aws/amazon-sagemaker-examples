class ModelPackageArnProvider:

  @staticmethod
  def get_construction_worker_model_package_arn(current_region):
      mapping = {
    "ap-south-1": "arn:aws:sagemaker:ap-south-1:077584701553:model-package/construction-worker-v1-copy-06-3f94f03fae021ca61cb609d42d0118c2",
    "ap-northeast-2": "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/construction-worker-v1-copy-06-3f94f03fae021ca61cb609d42d0118c2",
    "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/construction-worker-v1-copy-06-3f94f03fae021ca61cb609d42d0118c2",
    "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/construction-worker-v1-copy-06-3f94f03fae021ca61cb609d42d0118c2",
    "ap-northeast-1": "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/construction-worker-v1-copy-06-3f94f03fae021ca61cb609d42d0118c2",
    "ca-central-1": "arn:aws:sagemaker:ca-central-1:470592106596:model-package/construction-worker-v1-copy-06-3f94f03fae021ca61cb609d42d0118c2",
    "eu-central-1": "arn:aws:sagemaker:eu-central-1:446921602837:model-package/construction-worker-v1-copy-06-3f94f03fae021ca61cb609d42d0118c2",
    "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/construction-worker-v1-copy-06-3f94f03fae021ca61cb609d42d0118c2",
    "eu-west-2": "arn:aws:sagemaker:eu-west-2:856760150666:model-package/construction-worker-v1-copy-06-3f94f03fae021ca61cb609d42d0118c2",
    "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/construction-worker-v1-copy-06-3f94f03fae021ca61cb609d42d0118c2",
    "us-east-2": "arn:aws:sagemaker:us-west-1:382657785993:model-package/construction-worker-v1-copy-06-3f94f03fae021ca61cb609d42d0118c2",
    "us-west-1": "arn:aws:sagemaker:us-west-2:594846645681:model-package/construction-worker-v1-copy-06-3f94f03fae021ca61cb609d42d0118c2",
    "us-west-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/construction-worker-v1-copy-06-3f94f03fae021ca61cb609d42d0118c2"}
      return mapping[current_region]
  @staticmethod
  def get_machine_detection_model_package_arn(current_region):
        mapping = {
    "ap-northeast-1": "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/indus-construction-machines-eed6b262d4df3c8f46341abe757c5b63",
    "ap-northeast-2": "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/indus-construction-machines-eed6b262d4df3c8f46341abe757c5b63",
    "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/indus-construction-machines-eed6b262d4df3c8f46341abe757c5b63",
    "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/indus-construction-machines-eed6b262d4df3c8f46341abe757c5b63",
    "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/indus-construction-machines-eed6b262d4df3c8f46341abe757c5b63",
    "eu-central-1": "arn:aws:sagemaker:eu-central-1:446921602837:model-package/indus-construction-machines-eed6b262d4df3c8f46341abe757c5b63",
    "ap-south-1": "arn:aws:sagemaker:ap-south-1:077584701553:model-package/indus-construction-machines-eed6b262d4df3c8f46341abe757c5b63",
    "ca-central-1": "arn:aws:sagemaker:ca-central-1:470592106596:model-package/indus-construction-machines-eed6b262d4df3c8f46341abe757c5b63",
    "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/indus-construction-machines-eed6b262d4df3c8f46341abe757c5b63",
    "eu-west-2": "arn:aws:sagemaker:eu-west-2:856760150666:model-package/indus-construction-machines-eed6b262d4df3c8f46341abe757c5b63",
    "us-west-1": "arn:aws:sagemaker:us-west-1:382657785993:model-package/indus-construction-machines-eed6b262d4df3c8f46341abe757c5b63",
    "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/indus-construction-machines-eed6b262d4df3c8f46341abe757c5b63",
    "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/indus-construction-machines-eed6b262d4df3c8f46341abe757c5b63"}
        return mapping[current_region]

  @staticmethod
  def get_ppe_detection_model_package_arn(current_region):
      mapping = {
    "ap-south-1": "arn:aws:sagemaker:ap-south-1:077584701553:model-package/ppe-v1-copy-06-25-32446c1aac94cdb4e4d0e131f2efe62f",
    "ap-northeast-2": "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/ppe-v1-copy-06-25-32446c1aac94cdb4e4d0e131f2efe62f",
    "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/ppe-v1-copy-06-25-32446c1aac94cdb4e4d0e131f2efe62f",
    "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/ppe-v1-copy-06-25-32446c1aac94cdb4e4d0e131f2efe62f",
    "ap-northeast-1": "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/ppe-v1-copy-06-25-32446c1aac94cdb4e4d0e131f2efe62f",
    "ca-central-1": "arn:aws:sagemaker:ca-central-1:470592106596:model-package/ppe-v1-copy-06-25-32446c1aac94cdb4e4d0e131f2efe62f",
    "eu-central-1": "arn:aws:sagemaker:eu-central-1:446921602837:model-package/ppe-v1-copy-06-25-32446c1aac94cdb4e4d0e131f2efe62f",
    "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/ppe-v1-copy-06-25-32446c1aac94cdb4e4d0e131f2efe62f",
    "eu-west-2": "arn:aws:sagemaker:eu-west-2:856760150666:model-package/ppe-v1-copy-06-25-32446c1aac94cdb4e4d0e131f2efe62f",
    "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/ppe-v1-copy-06-25-32446c1aac94cdb4e4d0e131f2efe62f",
    "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/ppe-v1-copy-06-25-32446c1aac94cdb4e4d0e131f2efe62f",
    "us-west-1": "arn:aws:sagemaker:us-west-1:382657785993:model-package/ppe-v1-copy-06-25-32446c1aac94cdb4e4d0e131f2efe62f",
    "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/ppe-v1-copy-06-25-32446c1aac94cdb4e4d0e131f2efe62f"}
      return mapping[current_region]

  @staticmethod
  def get_hard_hat_detection_model_package_arn(current_region):
      mapping = {
    "ap-south-1": "arn:aws:sagemaker:ap-south-1:077584701553:model-package/hardhat-detection-gpu-2-e3449f86581997ece577e718d771238d",
    "ap-northeast-2": "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/hardhat-detection-gpu-2-e3449f86581997ece577e718d771238d",
    "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/hardhat-detection-gpu-2-e3449f86581997ece577e718d771238d",
    "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/hardhat-detection-gpu-2-e3449f86581997ece577e718d771238d",
    "ap-northeast-1": "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/hardhat-detection-gpu-2-e3449f86581997ece577e718d771238d",
    "ca-central-1": "arn:aws:sagemaker:ca-central-1:470592106596:model-package/hardhat-detection-gpu-2-e3449f86581997ece577e718d771238d",
    "eu-central-1": "arn:aws:sagemaker:eu-central-1:446921602837:model-package/hardhat-detection-gpu-2-e3449f86581997ece577e718d771238d",
    "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/hardhat-detection-gpu-2-e3449f86581997ece577e718d771238d",
    "eu-west-2": "arn:aws:sagemaker:eu-west-2:856760150666:model-package/hardhat-detection-gpu-2-e3449f86581997ece577e718d771238d",
    "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/hardhat-detection-gpu-2-e3449f86581997ece577e718d771238d",
    "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/hardhat-detection-gpu-2-e3449f86581997ece577e718d771238d",
    "us-west-1": "arn:aws:sagemaker:us-west-1:382657785993:model-package/hardhat-detection-gpu-2-e3449f86581997ece577e718d771238d",
    "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/hardhat-detection-gpu-2-e3449f86581997ece577e718d771238d"}
      return mapping[current_region]