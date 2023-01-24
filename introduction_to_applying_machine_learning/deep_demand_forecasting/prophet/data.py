from pathlib import Path

import pandas as pd
from typing import Tuple, Optional, List
from gluonts.dataset.rolling_dataset import StepStrategy, generate_rolling_dataset
from gluonts.dataset.common import load_datasets, ListDataset, TrainDatasets
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
    val_start_time, val_end_time = timestamps[-(len_single_ts_test-len_single_ts_train)], timestamps[-prediction_length]

    return (val_start_time, val_end_time)

def load_train_and_validation_datasets(
    path: Path, 
    prediction_length: int, 
    multiple_prediction_length_for_training: Optional[int] = 1) -> ListDataset:
    """
    Load train and validation time series data from file.

    Args:
        path: Path
            Path to the dataset.
            
        multiple_prediction_length_for_training: Optional[int]
            Select the length for training data.
        
        prediction_length: Optional[int]
            Select the prediction length.

    Returns:
        List of Train and validaion data in ListDataset format.
    """
    ds = load_datasets(path, path / "train", path / "test")
    target_dim = int(ds.metadata.feat_static_cat[0].cardinality)
    last_longest_batch_ts = list(ds.test)[-target_dim:]
    
    # generate rolling-based data
    val_start_time, val_end_time = find_validation_start_end_time(ds=ds, prediction_length=prediction_length)
    roll_val_list = generate_rolling_dataset(
        dataset=last_longest_batch_ts,
        strategy = StepStrategy(prediction_length=prediction_length, step_size=prediction_length),
        start_time = val_start_time,
        end_time = val_end_time,
    )
    ordered_roll_val_list = change_ts_order(roll_val_list)
    
    # group the same length of data together and create the list of ListDataset where 
    # each element within the ListDataset has the same start time.
    train_plus_validation_data = []
    group_data = []
    for idx, each_ts in enumerate(ordered_roll_val_list):
        if idx % target_dim == 0 and idx != 0:
            train_plus_validation_data.append(
                ListDataset(
                    group_data,
                    freq=ds.metadata.freq
                )
            )
            group_data = []
        current_length = each_ts['target'].shape[0]
        start_time = each_ts['start']
        timestamps = pd.date_range(start_time, periods=current_length, freq=ds.metadata.freq)
        group_data.append({
            "start": timestamps[-(multiple_prediction_length_for_training*prediction_length + prediction_length)],
            "target": each_ts["target"][-(prediction_length*multiple_prediction_length_for_training + prediction_length):]
        })
    train_plus_validation_data.append(
        ListDataset(
            group_data,
            freq=ds.metadata.freq
        )
    )
    return train_plus_validation_data
    
