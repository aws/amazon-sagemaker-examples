
class ModelPackageArnProvider:
    
    @staticmethod
    def get_vehicle_damage_detection_model_package_arn(current_region):
        mapping = {
    
          "us-east-1" : "arn:aws:sagemaker:us-east-1:865070037744:model-package/car-damage-prediction-modelpac-1c655c2ff7c7f4276b681db72abba25f",
          "ap-south-1" : "arn:aws:sagemaker:ap-south-1:077584701553:model-package/car-damage-prediction-modelpac-1c655c2ff7c7f4276b681db72abba25f",
          "ap-northeast-2" : "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/car-damage-prediction-modelpac-1c655c2ff7c7f4276b681db72abba25f",
          "ap-southeast-1" : "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/car-damage-prediction-modelpac-1c655c2ff7c7f4276b681db72abba25f",
          "ap-southeast-2" : "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/car-damage-prediction-modelpac-1c655c2ff7c7f4276b681db72abba25f",
          "ap-northeast-1" : "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/car-damage-prediction-modelpac-1c655c2ff7c7f4276b681db72abba25f",
          "ca-central-1" : "arn:aws:sagemaker:ca-central-1:470592106596:model-package/car-damage-prediction-modelpac-1c655c2ff7c7f4276b681db72abba25f",
          "eu-central-1" : "arn:aws:sagemaker:eu-central-1:446921602837:model-package/car-damage-prediction-modelpac-1c655c2ff7c7f4276b681db72abba25f",
          "eu-west-1" : "arn:aws:sagemaker:eu-west-1:985815980388:model-package/car-damage-prediction-modelpac-1c655c2ff7c7f4276b681db72abba25f",
          "eu-west-2" : "arn:aws:sagemaker:eu-west-2:856760150666:model-package/car-damage-prediction-modelpac-1c655c2ff7c7f4276b681db72abba25f",
          "us-east-2" : "arn:aws:sagemaker:us-east-2:057799348421:model-package/car-damage-prediction-modelpac-1c655c2ff7c7f4276b681db72abba25f",
           "us-west-1" : "arn:aws:sagemaker:us-west-1:382657785993:model-package/car-damage-prediction-modelpac-1c655c2ff7c7f4276b681db72abba25f",
            "us-west-2" : "arn:aws:sagemaker:us-west-2:594846645681:model-package/car-damage-prediction-modelpac-1c655c2ff7c7f4276b681db72abba25f"
                    }
        return mapping[current_region]
    

        
    @staticmethod
    def get_vehicle_recognition_model_package_arn(current_region):
        mapping = {
          "us-east-1" : "arn:aws:sagemaker:us-east-1:865070037744:model-package/vehicle-5bbb43353155de115c9fabdde5167c06",
          "ap-northeast-1" : "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/vehicle-5bbb43353155de115c9fabdde5167c06",
         "ap-northeast-2" : "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/vehicle-5bbb43353155de115c9fabdde5167c06",
          "ap-southeast-1" : "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/vehicle-5bbb43353155de115c9fabdde5167c06",
          "ap-southeast-2" : "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/vehicle-5bbb43353155de115c9fabdde5167c06",
          "eu-central-1" : "arn:aws:sagemaker:eu-central-1:446921602837:model-package/vehicle-5bbb43353155de115c9fabdde5167c06",
        "ap-south-1":    "arn:aws:sagemaker:ap-south-1:077584701553:model-package/vehicle-5bbb43353155de115c9fabdde5167c06",
         "ca-central-1":"arn:aws:sagemaker:ca-central-1:470592106596:model-package/vehicle-5bbb43353155de115c9fabdde5167c06",   
          "eu-west-1" : "arn:aws:sagemaker:eu-west-1:985815980388:model-package/vehicle-5bbb43353155de115c9fabdde5167c06",
          "eu-west-2" : "arn:aws:sagemaker:eu-west-2:856760150666:model-package/vehicle-5bbb43353155de115c9fabdde5167c06"
        }
        return mapping[current_region]