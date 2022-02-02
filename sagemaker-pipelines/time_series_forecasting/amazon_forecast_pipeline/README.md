# Amazon Forecast with SageMaker Pipelines

This SageMaker example showcases how you can create a dataset, dataset group and predictor with Amazon Forecast and SageMaker Pipelines.

## Contents

[sm_pipeline_with_amazon_forecast.ipynb](sm_pipeline_with_amazon_forecast.ipynb): Notebook explaining the pipeline step-by-step.

[preprocess.py](preprocess.py): Script used in the `ForecastPreProcess` step in pipeline for data preparation used for training and evaluation.

[train.py](train.py): Script used in `ForecastTrainAndEvaluate` step in pipeline to train and evaluate the Amazon
Forecast model.

[conditional_delete.py](conditional_delete.py): Script used in `ForecastCondtionalDelete` step in pipeline to delete all Forecast resources if the score achieved on a particular metric is not satisfactory.

[data](data): data folder containing the `train.csv`.
