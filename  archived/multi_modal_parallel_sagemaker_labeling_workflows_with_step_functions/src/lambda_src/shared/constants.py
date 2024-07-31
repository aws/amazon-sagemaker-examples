class SmgtJobType:
    BATCH = "BATCH"


class SMGTJobCategory:
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"


class BatchStatus:
    IN_PROGRESS = "IN_PROGRESS"
    VALIDATION_FAILURE = "VALIDATION_FAIL"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    WAIT_FOR_SMGT_RESPONSE = "WAIT_FOR_SMGT_RESPONSE"
    WAIT_FOR_METADATA_INPUT = "WAIT_FOR_META_DATA_INPUT"
    COMPLETE = "COMPLETE"


class BatchMetadataType:
    INPUT = "INPUT"
    VALIDATION = "VALIDATION"
    FIRST_LEVEL = "FIRST_LEVEL"
    SECOND_LEVEL = "SECOND_LEVEL"
    THIRD_LEVEL = "THIRD_LEVEL"
    FRAME_LEVEL = "FRAME_LEVEL"
    HUMAN_INPUT_METADATA = "HUMAN_INPUT_METADATA"
    PROCESS_LEVEL = "PROCESS_LEVEL"
    JOB_LEVEL = "JOB_LEVEL"


class BatchCurrentStep:
    INPUT = "INPUT"
    VALIDATION = "VALIDATION"
    FIRST_LEVEL = "FIRST_LEVEL"
    SECOND_LEVEL = "SECOND_LEVEL"
    THIRD_LEVEL = "THIRD_LEVEL"


class BatchMetadataTableAttributes:
    PARENT_BATCH_ID = "ParentBatchId"
    LABELING_JOBS = "LabelingJobs"
    BATCH_ID = "BatchId"
    BATCH_STATUS = "BatchStatus"
    BATCH_CURRENT_STEP = "BatchCurrentStep"
    BATCH_METADATA_TYPE = "BatchMetadataType"
    BATCH_EXECUTION_STEP = "BatchExecutionStep"
    LABELING_JOB_NAME = "LabelingJobName"
    LABEL_CATEGORY_CONFIG = "LabelCategoryConfig"
    LABEL_ATTRIBUTE_NAME = "LabelAttributeName"
    BATCH_LABELING_JOB_INPUT_DATA = "BatchLabelingJobInputData"

    NUM_CHILD_BATCHES = "NumChildBatches"
    NUM_CHILD_BATCHES_COMPLETE = "NumChildBatchesComplete"

    FIRST_LEVEL_BATCH_METADATA_ID = "FirstLevelBatchId"
    SECOND_LEVEL_BATCH_METADATA_ID = "SecondLevelBatchId"
    DOWN_SAMPLING_RATE = "DownSamplingRate"
    JOB_INPUT_LOCATION = "JobInputLocation"
    JOB_OUTPUT_LOCATION = "JobOutputLocation"
    JOB_DOWN_SAMPLE_LOCATION = "JobDownSampleLocation"
    MESSAGE = "Message"
    STATE_TOKEN = "StateToken"
    FRAME_INDEX = "FrameIndex"


SNS_DEDUPLICATION_KEY_NAME = "BatchFrameKey"
