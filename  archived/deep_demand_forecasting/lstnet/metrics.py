from typing import Dict

import numpy as np
from gluonts.dataset.common import ListDataset


def rrse(agg_metrics: Dict[str, float], dataset: ListDataset) -> np.ndarray:
    return agg_metrics["RMSE"] / np.std(next(iter(dataset))["target"])
