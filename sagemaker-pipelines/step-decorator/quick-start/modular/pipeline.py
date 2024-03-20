import os

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.function_step import step

from steps.preprocess import preprocess
from steps.train import train
from steps.evaluation import evaluate


def steps(
    input_data_path,
):
    data = step(preprocess, name="AbaloneProcess")(input_data_path)

    model = step(train, name="AbaloneTrain")(train_df=data[0], validation_df=data[1])

    evaluation_result = step(evaluate, name="AbaloneEval")(model=model, test_df=data[2])

    return [evaluation_result]


if __name__ == "__main__":
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()

    sagemaker_session = sagemaker.session.Session()

    input_path = (
        f"s3://sagemaker-example-files-prod-{sagemaker_session.boto_region_name}/datasets/tabular"
        f"/uci_abalone/abalone.csv"
    )

    steps = steps(
        input_data_path=input_path,
    )
    pipeline = Pipeline(
        name="AbalonePipelineModular",
        steps=steps,
    )
    # Note: sagemaker.get_execution_role does not work outside sagemaker
    pipeline.upsert(role_arn=sagemaker.get_execution_role())
    pipeline.start()
