from pathlib import Path

import pandas as pd
from typing import Tuple, Optional, List
from gluonts.dataset.rolling_dataset import StepStrategy, generate_rolling_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.common import TrainDatasets, load_datasets, ListDataset
from collections import defaultdict

def change_ts_order(roll_val_list: list) -> List:
    """
    Sort the list of time series according to the dimension of the elements in ascending order.

    Args:
        roll_val_list: list:
            List of input data.

    Returns:
        List of the sorted elements.
    """
    groups = defaultdict(list)
    for obj in roll_val_list:
        groups[obj['target'].shape[0]].append(obj)
    
    ordered_roll_val_list = []
    for k, v in groups.items():
        ordered_roll_val_list.extend(v)

    return ordered_roll_val_list

def find_validation_start_end_time(ds: TrainDatasets, prediction_length: int) -> Tuple[pd.core.indexes.datetimes.DatetimeIndex, pd.core.indexes.datetimes.DatetimeIndex]:
    """
    Find validation start time and end time of the given test data.
    The validation start time is right after the training end time and
    the validation end time is the time at the end of testing data minus the prediction length.

    Args:
        ds: TrainDatasets:
            Input data.
            
        prediction_length: int
            Prediction length.

    Returns:
        Tuple of start time and end time.
    """
    first_ts_train, last_ts_test = next(iter(ds.train)), list(ds.test)[-1]
    len_single_ts_train, len_single_ts_test = len(first_ts_train['target']), len(last_ts_test['target'])

    start_time = last_ts_test['start']
    timestamps = pd.date_range(start_time, periods=len_single_ts_test, freq=ds.metadata.freq)
    val_start_time, val_end_time = timestamps[-(len_single_ts_test-len_single_ts_train)], timestamps[-prediction_length-1]

    return (val_start_time, val_end_time)


def load_multivariate_datasets(path: Path, is_validation: bool = False, prediction_length: Optional[int] = None, load_raw: Optional[bool] = False) -> TrainDatasets:
    """
    Load multivariate time series data from file.

    Args:
        path: Path
            Path to the dataset.
            
        is_validation: bool
            Whether to select the validation or test data.
        
        prediction_length: Optional[int]
            Required if is_validation is specified.

    Returns:
        Train and test or validaion data in TrainDatasets format.
    """
    metadata_path = path if (load_raw or path == Path("raw_data")) else path / "metadata"
    ds = load_datasets(metadata_path, path / "train", path / "test")
    target_dim = int(ds.metadata.feat_static_cat[0].cardinality)
    last_longest_batch_ts = list(ds.test)[-target_dim:]
    
    grouper_train = MultivariateGrouper(max_target_dim=target_dim)
    grouper_train_data = grouper_train(ds.train)
    if is_validation:
        assert prediction_length is not None, "argument 'prediction_length' is required to create a validation data. Please specify the 'prediction_length'."
        val_start_time, val_end_time = find_validation_start_end_time(ds=ds, prediction_length=prediction_length)
        roll_val_list = generate_rolling_dataset(
            dataset=last_longest_batch_ts,
            strategy = StepStrategy(prediction_length=prediction_length, step_size=prediction_length),
            start_time = val_start_time,
            end_time = val_end_time,
        )
        ordered_roll_val_list = change_ts_order(roll_val_list)
        grouper_test = MultivariateGrouper(max_target_dim=target_dim, num_test_dates=len(ordered_roll_val_list)/target_dim)
        grouper_test_data = grouper_test(ListDataset(data_iter=ordered_roll_val_list, freq=ds.metadata.freq))
    else:
        grouper_test = MultivariateGrouper(max_target_dim=target_dim)
        grouper_test_data = grouper_test(ds.test)
        
    return TrainDatasets(
        metadata=ds.metadata,
        train=grouper_train_data,
        test=grouper_test_data,
    )
