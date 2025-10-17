from datetime import datetime
import numpy as np
import torch
from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.model_functional import FunctionalModel


@model_metric(name="accuracy")
def accuracy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: FunctionalModel,
) -> list[Measure]:
    # print(f"Dataset size: {len(dataset.data)}")

    # target column name of datashape
    target_col = datashape.target.name

    # feature column names of datashape
    feature_cols = [feature.name for feature in datashape.features]

    # extract features X & target y from the dataset
    X = dataset.data[feature_cols]
    y = dataset.data[target_col]

    # convert the features to tensor using PyTorch
    x_tensor = torch.tensor(X.values, dtype=torch.float32)

    # get predictions from the funct. model
    predictions = functional_model.predict(x_tensor)

    # calculate accuracy
    correct_predictions = np.sum(predictions == y)
    total_predictions = len(y)

    accuracy_value = correct_predictions / total_predictions

    current_time = datetime.now()

    return [Measure(name="accuracy", score=accuracy_value, time=current_time)]
