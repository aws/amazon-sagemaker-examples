"""
Helpers for getting PRE and POST annotation ARNs for labeling jobs.
"""

from enum import Enum


class JobModality(str, Enum):
    PointCloudObjectDetection = "PointCloudObjectDetection"
    PointCloudObjectDetectionAudit = "PointCloudObjectDetectionAudit"
    PointCloudObjectTracking = "PointCloudObjectTracking"
    PointCloudObjectTrackingAudit = "PointCloudObjectTrackingAudit"
    PointCloudSemanticSegmentation = "PointCloudSemanticSegmentation"
    PointCloudSemanticSegmentationAudit = "PointCloudSemanticSegmentationAudit"
    VideoObjectDetection = "VideoObjectDetection"
    VideoObjectDetectionAudit = "VideoObjectDetectionAudit"
    VideoObjectTracking = "VideoObjectTracking"
    VideoObjectTrackingAudit = "VideoObjectTrackingAudit"

    def is_member(job_type):
        return job_type in JobModality.__members__

    def job_name_to_label_attribute(job_type, name):
        """Converts a job name to a label attribute value"""
        if job_type.startswith("Video"):
            return f"{name}-ref"
        return name


def ui_config(region, job_type):
    """Generates a ui_config for a supported task type."""

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_labeling_job
    human_task_ui_arns = {
        JobModality.PointCloudObjectDetection: f"arn:aws:sagemaker:{region}:394669845002:human-task-ui/PointCloudObjectDetection",
        JobModality.PointCloudObjectDetectionAudit: f"arn:aws:sagemaker:{region}:394669845002:human-task-ui/PointCloudObjectDetection",
        JobModality.PointCloudObjectTracking: f"arn:aws:sagemaker:{region}:394669845002:human-task-ui/PointCloudObjectTracking",
        JobModality.PointCloudObjectTrackingAudit: f"arn:aws:sagemaker:{region}:394669845002:human-task-ui/PointCloudObjectTracking",
        JobModality.PointCloudSemanticSegmentation: f"arn:aws:sagemaker:{region}:394669845002:human-task-ui/PointCloudSemanticSegmentation",
        JobModality.PointCloudSemanticSegmentationAudit: f"arn:aws:sagemaker:{region}:394669845002:human-task-ui/PointCloudSemanticSegmentation",
        JobModality.VideoObjectDetection: f"arn:aws:sagemaker:{region}:394669845002:human-task-ui/VideoObjectDetection",
        JobModality.VideoObjectDetectionAudit: f"arn:aws:sagemaker:{region}:394669845002:human-task-ui/VideoObjectDetection",
        JobModality.VideoObjectTracking: f"arn:aws:sagemaker:{region}:394669845002:human-task-ui/VideoObjectTracking",
        JobModality.VideoObjectTrackingAudit: f"arn:aws:sagemaker:{region}:394669845002:human-task-ui/VideoObjectTracking",
    }

    human_task_arn = human_task_ui_arns[job_type]

    return {
        "HumanTaskUiArn": human_task_arn,
    }


def pre_human_task_lambda_arn(region, job_type):
    """Generates a pre human task lambda arn for a supported task type."""
    pre_human_task_lambdas = {
        JobModality.PointCloudObjectDetection: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:PRE-3DPointCloudObjectDetection",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:PRE-3DPointCloudObjectDetection",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:PRE-3DPointCloudObjectDetection",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:PRE-3DPointCloudObjectDetection",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-3DPointCloudObjectDetection",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-3DPointCloudObjectDetection",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:PRE-3DPointCloudObjectDetection",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:PRE-3DPointCloudObjectDetection",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-3DPointCloudObjectDetection",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:PRE-3DPointCloudObjectDetection",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-3DPointCloudObjectDetection",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:PRE-3DPointCloudObjectDetection",
        },
        JobModality.PointCloudObjectDetectionAudit: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:PRE-Adjustment3DPointCloudObjectDetection",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:PRE-Adjustment3DPointCloudObjectDetection",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:PRE-Adjustment3DPointCloudObjectDetection",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:PRE-Adjustment3DPointCloudObjectDetection",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-Adjustment3DPointCloudObjectDetection",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-Adjustment3DPointCloudObjectDetection",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:PRE-Adjustment3DPointCloudObjectDetection",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:PRE-Adjustment3DPointCloudObjectDetection",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-Adjustment3DPointCloudObjectDetection",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:PRE-Adjustment3DPointCloudObjectDetection",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-Adjustment3DPointCloudObjectDetection",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:PRE-Adjustment3DPointCloudObjectDetection",
        },
        JobModality.PointCloudObjectTracking: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:PRE-3DPointCloudObjectTracking",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:PRE-3DPointCloudObjectTracking",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:PRE-3DPointCloudObjectTracking",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:PRE-3DPointCloudObjectTracking",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-3DPointCloudObjectTracking",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-3DPointCloudObjectTracking",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:PRE-3DPointCloudObjectTracking",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:PRE-3DPointCloudObjectTracking",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-3DPointCloudObjectTracking",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:PRE-3DPointCloudObjectTracking",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-3DPointCloudObjectTracking",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:PRE-3DPointCloudObjectTracking",
        },
        JobModality.PointCloudObjectTrackingAudit: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:PRE-Adjustment3DPointCloudObjectTracking",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:PRE-Adjustment3DPointCloudObjectTracking",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:PRE-Adjustment3DPointCloudObjectTracking",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:PRE-Adjustment3DPointCloudObjectTracking",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-Adjustment3DPointCloudObjectTracking",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-Adjustment3DPointCloudObjectTracking",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:PRE-Adjustment3DPointCloudObjectTracking",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:PRE-Adjustment3DPointCloudObjectTracking",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-Adjustment3DPointCloudObjectTracking",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:PRE-Adjustment3DPointCloudObjectTracking",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-Adjustment3DPointCloudObjectTracking",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:PRE-Adjustment3DPointCloudObjectTracking",
        },
        JobModality.PointCloudSemanticSegmentation: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:PRE-3DPointCloudSemanticSegmentation",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:PRE-3DPointCloudSemanticSegmentation",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:PRE-3DPointCloudSemanticSegmentation",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:PRE-3DPointCloudSemanticSegmentation",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-3DPointCloudSemanticSegmentation",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-3DPointCloudSemanticSegmentation",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:PRE-3DPointCloudSemanticSegmentation",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:PRE-3DPointCloudSemanticSegmentation",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-3DPointCloudSemanticSegmentation",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:PRE-3DPointCloudSemanticSegmentation",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-3DPointCloudSemanticSegmentation",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:PRE-3DPointCloudSemanticSegmentation",
        },
        JobModality.PointCloudSemanticSegmentationAudit: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:PRE-Adjustment3DPointCloudSemanticSegmentation",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:PRE-Adjustment3DPointCloudSemanticSegmentation",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:PRE-Adjustment3DPointCloudSemanticSegmentation",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:PRE-Adjustment3DPointCloudSemanticSegmentation",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-Adjustment3DPointCloudSemanticSegmentation",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-Adjustment3DPointCloudSemanticSegmentation",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:PRE-Adjustment3DPointCloudSemanticSegmentation",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:PRE-Adjustment3DPointCloudSemanticSegmentation",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-Adjustment3DPointCloudSemanticSegmentation",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:PRE-Adjustment3DPointCloudSemanticSegmentation",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-Adjustment3DPointCloudSemanticSegmentation",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:PRE-Adjustment3DPointCloudSemanticSegmentation",
        },
        JobModality.VideoObjectDetection: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:PRE-VideoObjectDetection",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:PRE-VideoObjectDetection",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:PRE-VideoObjectDetection",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:PRE-VideoObjectDetection",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-VideoObjectDetection",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-VideoObjectDetection",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:PRE-VideoObjectDetection",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:PRE-VideoObjectDetection",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-VideoObjectDetection",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:PRE-VideoObjectDetection",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-VideoObjectDetection",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:PRE-VideoObjectDetection",
        },
        JobModality.VideoObjectDetectionAudit: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:PRE-AdjustmentVideoObjectDetection",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:PRE-AdjustmentVideoObjectDetection",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:PRE-AdjustmentVideoObjectDetection",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:PRE-AdjustmentVideoObjectDetection",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-AdjustmentVideoObjectDetection",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-AdjustmentVideoObjectDetection",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:PRE-AdjustmentVideoObjectDetection",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:PRE-AdjustmentVideoObjectDetection",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-AdjustmentVideoObjectDetection",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:PRE-AdjustmentVideoObjectDetection",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-AdjustmentVideoObjectDetection",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:PRE-AdjustmentVideoObjectDetection",
        },
        JobModality.VideoObjectTracking: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:PRE-VideoObjectTracking",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:PRE-VideoObjectTracking",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:PRE-VideoObjectTracking",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:PRE-VideoObjectTracking",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-VideoObjectTracking",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-VideoObjectTracking",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:PRE-VideoObjectTracking",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:PRE-VideoObjectTracking",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-VideoObjectTracking",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:PRE-VideoObjectTracking",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-VideoObjectTracking",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:PRE-VideoObjectTracking",
        },
        JobModality.VideoObjectTrackingAudit: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:PRE-AdjustmentVideoObjectTracking",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:PRE-AdjustmentVideoObjectTracking",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:PRE-AdjustmentVideoObjectTracking",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:PRE-AdjustmentVideoObjectTracking",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:PRE-AdjustmentVideoObjectTracking",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:PRE-AdjustmentVideoObjectTracking",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:PRE-AdjustmentVideoObjectTracking",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:PRE-AdjustmentVideoObjectTracking",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:PRE-AdjustmentVideoObjectTracking",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:PRE-AdjustmentVideoObjectTracking",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:PRE-AdjustmentVideoObjectTracking",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:PRE-AdjustmentVideoObjectTracking",
        },
    }
    return pre_human_task_lambdas[job_type][region]


def annotation_consolidation_config(region, job_type):
    annotation_consolidation_lambda_arns = {
        JobModality.PointCloudObjectDetection: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:ACS-3DPointCloudObjectDetection",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:ACS-3DPointCloudObjectDetection",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:ACS-3DPointCloudObjectDetection",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:ACS-3DPointCloudObjectDetection",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-3DPointCloudObjectDetection",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-3DPointCloudObjectDetection",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:ACS-3DPointCloudObjectDetection",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:ACS-3DPointCloudObjectDetection",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-3DPointCloudObjectDetection",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:ACS-3DPointCloudObjectDetection",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-3DPointCloudObjectDetection",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:ACS-3DPointCloudObjectDetection",
        },
        JobModality.PointCloudObjectDetectionAudit: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:ACS-Adjustment3DPointCloudObjectDetection",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:ACS-Adjustment3DPointCloudObjectDetection",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:ACS-Adjustment3DPointCloudObjectDetection",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:ACS-Adjustment3DPointCloudObjectDetection",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-Adjustment3DPointCloudObjectDetection",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-Adjustment3DPointCloudObjectDetection",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:ACS-Adjustment3DPointCloudObjectDetection",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:ACS-Adjustment3DPointCloudObjectDetection",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-Adjustment3DPointCloudObjectDetection",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:ACS-Adjustment3DPointCloudObjectDetection",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-Adjustment3DPointCloudObjectDetection",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:ACS-Adjustment3DPointCloudObjectDetection",
        },
        JobModality.PointCloudObjectTracking: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:ACS-3DPointCloudObjectTracking",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:ACS-3DPointCloudObjectTracking",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:ACS-3DPointCloudObjectTracking",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:ACS-3DPointCloudObjectTracking",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-3DPointCloudObjectTracking",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-3DPointCloudObjectTracking",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:ACS-3DPointCloudObjectTracking",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:ACS-3DPointCloudObjectTracking",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-3DPointCloudObjectTracking",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:ACS-3DPointCloudObjectTracking",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-3DPointCloudObjectTracking",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:ACS-3DPointCloudObjectTracking",
        },
        JobModality.PointCloudObjectTrackingAudit: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:ACS-Adjustment3DPointCloudObjectTracking",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:ACS-Adjustment3DPointCloudObjectTracking",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:ACS-Adjustment3DPointCloudObjectTracking",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:ACS-Adjustment3DPointCloudObjectTracking",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-Adjustment3DPointCloudObjectTracking",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-Adjustment3DPointCloudObjectTracking",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:ACS-Adjustment3DPointCloudObjectTracking",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:ACS-Adjustment3DPointCloudObjectTracking",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-Adjustment3DPointCloudObjectTracking",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:ACS-Adjustment3DPointCloudObjectTracking",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-Adjustment3DPointCloudObjectTracking",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:ACS-Adjustment3DPointCloudObjectTracking",
        },
        JobModality.PointCloudSemanticSegmentation: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:ACS-3DPointCloudSemanticSegmentation",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:ACS-3DPointCloudSemanticSegmentation",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:ACS-3DPointCloudSemanticSegmentation",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:ACS-3DPointCloudSemanticSegmentation",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-3DPointCloudSemanticSegmentation",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-3DPointCloudSemanticSegmentation",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:ACS-3DPointCloudSemanticSegmentation",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:ACS-3DPointCloudSemanticSegmentation",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-3DPointCloudSemanticSegmentation",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:ACS-3DPointCloudSemanticSegmentation",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-3DPointCloudSemanticSegmentation",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:ACS-3DPointCloudSemanticSegmentation",
        },
        JobModality.PointCloudSemanticSegmentationAudit: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:ACS-3DPointCloudSemanticSegmentation",
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:ACS-Adjustment3DPointCloudSemanticSegmentation",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:ACS-Adjustment3DPointCloudSemanticSegmentation",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:ACS-Adjustment3DPointCloudSemanticSegmentation",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:ACS-Adjustment3DPointCloudSemanticSegmentation",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-Adjustment3DPointCloudSemanticSegmentation",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-Adjustment3DPointCloudSemanticSegmentation",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:ACS-Adjustment3DPointCloudSemanticSegmentation",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:ACS-Adjustment3DPointCloudSemanticSegmentation",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-Adjustment3DPointCloudSemanticSegmentation",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:ACS-Adjustment3DPointCloudSemanticSegmentation",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-Adjustment3DPointCloudSemanticSegmentation",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:ACS-Adjustment3DPointCloudSemanticSegmentation",
        },
        JobModality.VideoObjectDetection: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:ACS-VideoObjectDetection",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:ACS-VideoObjectDetection",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:ACS-VideoObjectDetection",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:ACS-VideoObjectDetection",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-VideoObjectDetection",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-VideoObjectDetection",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:ACS-VideoObjectDetection",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:ACS-VideoObjectDetection",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-VideoObjectDetection",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:ACS-VideoObjectDetection",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-VideoObjectDetection",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:ACS-VideoObjectDetection",
        },
        JobModality.VideoObjectDetectionAudit: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:ACS-AdjustmentVideoObjectDetection",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:ACS-AdjustmentVideoObjectDetection",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:ACS-AdjustmentVideoObjectDetection",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:ACS-AdjustmentVideoObjectDetection",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-AdjustmentVideoObjectDetection",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-AdjustmentVideoObjectDetection",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:ACS-AdjustmentVideoObjectDetection",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:ACS-AdjustmentVideoObjectDetection",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-AdjustmentVideoObjectDetection",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:ACS-AdjustmentVideoObjectDetection",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-AdjustmentVideoObjectDetection",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:ACS-AdjustmentVideoObjectDetection",
        },
        JobModality.VideoObjectTracking: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:ACS-VideoObjectTracking",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:ACS-VideoObjectTracking",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:ACS-VideoObjectTracking",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:ACS-VideoObjectTracking",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-VideoObjectTracking",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-VideoObjectTracking",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:ACS-VideoObjectTracking",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:ACS-VideoObjectTracking",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-VideoObjectTracking",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:ACS-VideoObjectTracking",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-VideoObjectTracking",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:ACS-VideoObjectTracking",
        },
        JobModality.VideoObjectTrackingAudit: {
            "us-east-1": "arn:aws:lambda:us-east-1:432418664414:function:ACS-AdjustmentVideoObjectTracking",
            "us-east-2": "arn:aws:lambda:us-east-2:266458841044:function:ACS-AdjustmentVideoObjectTracking",
            "us-west-2": "arn:aws:lambda:us-west-2:081040173940:function:ACS-AdjustmentVideoObjectTracking",
            "eu-west-1": "arn:aws:lambda:eu-west-1:568282634449:function:ACS-AdjustmentVideoObjectTracking",
            "ap-northeast-1": "arn:aws:lambda:ap-northeast-1:477331159723:function:ACS-AdjustmentVideoObjectTracking",
            "ap-southeast-2": "arn:aws:lambda:ap-southeast-2:454466003867:function:ACS-AdjustmentVideoObjectTracking",
            "ap-south-1": "arn:aws:lambda:ap-south-1:565803892007:function:ACS-AdjustmentVideoObjectTracking",
            "eu-central-1": "arn:aws:lambda:eu-central-1:203001061592:function:ACS-AdjustmentVideoObjectTracking",
            "ap-northeast-2": "arn:aws:lambda:ap-northeast-2:845288260483:function:ACS-AdjustmentVideoObjectTracking",
            "eu-west-2": "arn:aws:lambda:eu-west-2:487402164563:function:ACS-AdjustmentVideoObjectTracking",
            "ap-southeast-1": "arn:aws:lambda:ap-southeast-1:377565633583:function:ACS-AdjustmentVideoObjectTracking",
            "ca-central-1": "arn:aws:lambda:ca-central-1:918755190332:function:ACS-AdjustmentVideoObjectTracking",
        },
    }
    return {
        "AnnotationConsolidationLambdaArn": annotation_consolidation_lambda_arns[job_type][region]
    }
