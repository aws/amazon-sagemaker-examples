class TestContext:
    test_context = "fake_context_object"
    invoked_function_arn = "test"


class OutputTestData:

    input_batch_to_human_readable_output = {
        "batchId": "batch-test-non-streaming-13",
        "status": "COMPLETE",
        "inputLabelingJobs": [
            {
                "inputConfig": {"inputManifestS3Uri": "test"},
                "jobLevel": 1,
                "jobModality": "PointCloudObjectDetectionAudit",
                "jobName": "batch-test-non-streaming-11-first",
                "jobType": "BATCH",
                "labelCategoryConfigS3Uri": "test",
                "maxConcurrentTaskCount": 100,
                "taskAvailabilityLifetimeInSeconds": 864000,
                "taskTimeLimitInSeconds": 604800,
                "workteamArn": "test",
            },
            {
                "inputConfig": {"chainFromJobName": "batch-test-non-streaming-11-first"},
                "jobLevel": 2,
                "jobModality": "PointCloudObjectDetectionAudit",
                "jobName": "batch-test-non-streaming-11-second",
                "jobType": "BATCH",
                "maxConcurrentTaskCount": 100,
                "taskAvailabilityLifetimeInSeconds": 864000,
                "taskTimeLimitInSeconds": 604800,
                "workteamArn": "atest",
            },
            {
                "inputConfig": {"chainFromJobName": "batch-test-non-streaming-11-second"},
                "jobLevel": 3,
                "jobModality": "PointCloudObjectDetectionAudit",
                "jobName": "batch-test-non-streaming-11-third",
                "jobType": "BATCH",
                "maxConcurrentTaskCount": 100,
                "taskAvailabilityLifetimeInSeconds": 864000,
                "taskTimeLimitInSeconds": 604800,
                "workteamArn": "test",
            },
        ],
        "firstLevel": {
            "status": "COMPLETE",
            "numChildBatches": 1,
            "numChildBatchesComplete": 1,
            "jobLevels": [
                {
                    "batchId": "12345",
                    "batchStatus": "COMPLETE",
                    "labelingJobName": "batch-test-non-streaming-11-second",
                    "labelAttributeName": "batch-test-non-streaming-11-second",
                    "labelCategoryS3Uri": "s3://testbucket/category-file.json",
                    "jobInputS3Uri": "s3://testbucket/category-file.json",
                    "jobInputS3Url": "https://testbucket.s3.amazonaws.com/category-file.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAY5FHYJAYKV7YHGUP%2F20210416%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210416T160942Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4045fba703131d01b8b7c2ad4fcfde9b34072e76200eb38b6eedcca6db5b6d72",
                    "jobOutputS3Uri": "s3://testbucket/category-file.json",
                    "jobOutputS3Url": "https://testbucket.s3.amazonaws.com/category-file.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAY5FHYJAYKV7YHGUP%2F20210416%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210416T160942Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4045fba703131d01b8b7c2ad4fcfde9b34072e76200eb38b6eedcca6db5b6d72",
                    "numFrames": 1,
                    "numFramesCompleted": 1,
                }
            ],
        },
        "secondLevel": {
            "status": "COMPLETE",
            "numChildBatches": 1,
            "numChildBatchesComplete": 1,
            "jobLevels": [
                {
                    "batchId": "12345",
                    "batchStatus": "COMPLETE",
                    "labelingJobName": "batch-test-non-streaming-11-second",
                    "labelAttributeName": "batch-test-non-streaming-11-second",
                    "labelCategoryS3Uri": "s3://testbucket/category-file.json",
                    "jobInputS3Uri": "s3://testbucket/category-file.json",
                    "jobInputS3Url": "https://testbucket.s3.amazonaws.com/category-file.json?X-Amz-Algorithm=AWS4"
                    "-HMAC-SHA256&X-Amz-Credential=AKIAY5FHYJAYKV7YHGUP%2F20210416%2Fus-west-2%2Fs3"
                    "%2Faws4_request&X-Amz-Date=20210416T160942Z&X-Amz-Expires=86400&X-Amz"
                    "-SignedHeaders=host&X-Amz-Signature"
                    "=4045fba703131d01b8b7c2ad4fcfde9b34072e76200eb38b6eedcca6db5b6d72",
                    "jobOutputS3Uri": "s3://testbucket/category-file.json",
                    "jobOutputS3Url": "https://testbucket.s3.amazonaws.com/category-file.json?X-Amz-Algorithm=AWS4"
                    "-HMAC-SHA256&X-Amz-Credential=AKIAY5FHYJAYKV7YHGUP%2F20210416%2Fus-west-2%2Fs3"
                    "%2Faws4_request&X-Amz-Date=20210416T160942Z&X-Amz-Expires=86400&X-Amz"
                    "-SignedHeaders=host&X-Amz-Signature"
                    "=4045fba703131d01b8b7c2ad4fcfde9b34072e76200eb38b6eedcca6db5b6d72",
                    "numFrames": 1,
                    "numFramesCompleted": 1,
                }
            ],
        },
        "thirdLevel": {
            "status": "COMPLETE",
            "numChildBatches": 1,
            "numChildBatchesComplete": 1,
            "jobLevels": [
                {
                    "batchId": "12345",
                    "batchStatus": "COMPLETE",
                    "labelingJobName": "batch-test-non-streaming-11-second",
                    "labelAttributeName": "batch-test-non-streaming-11-second",
                    "labelCategoryS3Uri": "s3://testbucket/category-file.json",
                    "jobInputS3Uri": "s3://testbucket/category-file.json",
                    "jobInputS3Url": "https://testbucket.s3.amazonaws.com/category-file.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAY5FHYJAYKV7YHGUP%2F20210416%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210416T160942Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4045fba703131d01b8b7c2ad4fcfde9b34072e76200eb38b6eedcca6db5b6d72",
                    "jobOutputS3Uri": "s3://testbucket/category-file.json",
                    "jobOutputS3Url": "https://testbucket.s3.amazonaws.com/category-file.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAY5FHYJAYKV7YHGUP%2F20210416%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210416T160942Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4045fba703131d01b8b7c2ad4fcfde9b34072e76200eb38b6eedcca6db5b6d72",
                    "numFrames": 1,
                    "numFramesCompleted": 1,
                }
            ],
        },
    }

    get_child_batch_metadata_output = [
        {
            "BatchId": "12345",
            "BatchMetadataType": "JOB_LEVEL",
            "NumChildBatches": 1,
            "NumChildBatchesComplete": 1,
            "BatchStatus": "COMPLETE",
            "JobInputLocation": "s3://testbucket/category-file.json",
            "JobOutputLocation": "s3://testbucket/category-file.json",
            "LabelAttributeName": "batch-test-non-streaming-11-second",
            "LabelCategoryConfig": "s3://testbucket/category-file.json",
            "LabelingJobName": "batch-test-non-streaming-11-second",
            "Message": "",
            "ParentBatchId": "batch-test-non-streaming-13-second_level",
        }
    ]

    get_batch_metadata_output = {
        "BatchCurrentStep": "INPUT",
        "BatchId": "batch-test-non-streaming-13",
        "BatchMetadataType": "INPUT",
        "BatchStatus": "COMPLETE",
        "LabelingJobs": [
            {
                "inputConfig": {"inputManifestS3Uri": "test"},
                "jobLevel": 1,
                "jobModality": "PointCloudObjectDetectionAudit",
                "jobName": "batch-test-non-streaming-11-first",
                "jobType": "BATCH",
                "labelCategoryConfigS3Uri": "test",
                "maxConcurrentTaskCount": 100,
                "taskAvailabilityLifetimeInSeconds": 864000,
                "taskTimeLimitInSeconds": 604800,
                "workteamArn": "test",
            },
            {
                "inputConfig": {
                    "chainFromJobName": "batch-test-non-streaming-11-first",
                    "downSamplingRate": 50,
                },
                "jobLevel": 2,
                "jobModality": "PointCloudObjectDetectionAudit",
                "jobName": "batch-test-non-streaming-11-second",
                "jobType": "BATCH",
                "maxConcurrentTaskCount": 100,
                "taskAvailabilityLifetimeInSeconds": 864000,
                "taskTimeLimitInSeconds": 604800,
                "workteamArn": "atest",
            },
            {
                "inputConfig": {"chainFromJobName": "batch-test-non-streaming-11-second"},
                "jobLevel": 3,
                "jobModality": "PointCloudObjectDetectionAudit",
                "jobName": "batch-test-non-streaming-11-third",
                "jobType": "BATCH",
                "maxConcurrentTaskCount": 100,
                "taskAvailabilityLifetimeInSeconds": 864000,
                "taskTimeLimitInSeconds": 604800,
                "workteamArn": "test",
            },
        ],
        "Message": "",
    }

    get_batch_first_level_output = {
        "BatchId": "first_level-batch",
        "BatchMetadataType": "FIRST_LEVEL",
        "BatchStatus": "IN_PROGRESS",
        "NumChildBatches": 2,
        "NumChildBatchesComplete": 0,
        "ParentBatchId": "parent-batch",
        "StateToken": "test",
    }

    get_batch_metadata_by_labeling_job_name_output = [
        {
            "BatchId": "batch-test-non-streaming-13-second_level-batch-test-non-streaming-11-second",
            "BatchMetadataType": "JOB_LEVEL",
            "BatchStatus": "COMPLETE",
            "JobInputLocation": "s3://test/test",
            "JobOutputLocation": "s3://test/test",
            "LabelAttributeName": "batch-test-non-streaming-11-second",
            "LabelCategoryConfig": "s3://test",
            "LabelingJobName": "batch-test-non-streaming-11-second",
            "Message": "",
            "ParentBatchId": "batch-test-non-streaming-13-second_level",
        }
    ]


class InputTestData:
    show_batch_request = {"batchId": ["batch-test-non-streaming-11"]}

    create_batch_request = {
        "batchId": "batch-test-non-streaming-11",
        "downSamplingRate": 50,
        "labelingJobs": [
            {
                "jobName": "batch-test-non-streaming-11-first",
                "jobType": "BATCH",
                "jobModality": "PointCloudObjectDetectionAudit",
                "labelCategoryConfigS3Uri": "s3://{{S3InputBucket}}/first-level-label-category-file.json",
                "inputConfig": {
                    "inputManifestS3Uri": "s3://{{S3InputBucket}}/two-frame-test.manifest"
                },
                "jobLevel": 1,
            },
            {
                "jobName": "batch-test-non-streaming-11-second",
                "jobType": "BATCH",
                "jobModality": "PointCloudObjectDetectionAudit",
                "inputConfig": {"chainFromJobName": "batch-test-non-streaming-11-first"},
                "jobLevel": 2,
            },
            {
                "jobName": "batch-test-non-streaming-11-third",
                "jobType": "BATCH",
                "jobModality": "PointCloudObjectDetectionAudit",
                "inputConfig": {"chainFromJobName": "batch-test-non-streaming-11-second"},
                "jobLevel": 3,
            },
        ],
    }
