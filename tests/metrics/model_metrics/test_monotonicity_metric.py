import datetime
import uuid

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from tests.save_measures_utils import save_measures

from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Model,
    FeatureType,
    Feature,
)
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric_registry
from a4s_eval.metric_registries.model_metric_registry import ModelMetric
from a4s_eval.metrics.model_metrics.monotonicity_cifar10 import monotonicity_cifar10


def _load_cifar10_model() -> nn.Module:
    """
    Loads a PyTorch ResNet-18 model structure (CIFAR-10).
    """
    model = models.resnet18(weights=None)

    # modify for CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)

    # load pretrained CIFAR-10 checkpoint
    checkpoint_url = "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin"
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model


@pytest.fixture
def data_shape() -> DataShape:
    target = Feature(
        pid=uuid.uuid4(),
        name="target",
        feature_type=FeatureType.INTEGER,
        min_value=0,  # CIFAR-10 has 10 classes
        max_value=9,
    )

    image_feature = Feature(
        pid=uuid.uuid4(),
        name="image",
        feature_type=FeatureType.FLOAT,
        min_value=0.0,
        max_value=1.0,
    )

    datashape = DataShape(features=[image_feature], date=None, target=target)

    return datashape


@pytest.fixture
def ref_model(test_dataset: Dataset) -> Model:
    return _load_cifar10_model()


@pytest.fixture
def test_dataset(data_shape: DataShape) -> Dataset:
    # setup data transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_set = datasets.CIFAR10(
        root="./tests/data", train=False, download=True, transform=transform
    )

    # extract 3 samples for testing
    num_samples = 3
    test_indices = list(range(num_samples))
    subset = torch.utils.data.Subset(test_set, test_indices)

    # extract tensors
    data_loader = torch.utils.data.DataLoader(subset, batch_size=num_samples)
    x_tensor, y_labels_tensor = next(iter(data_loader))

    # convert to numpy and create dataframe
    y_labels = y_labels_tensor.cpu().numpy()

    data = pd.DataFrame(
        {
            "image": [x_tensor[i] for i in range(len(x_tensor))],
            "target": y_labels,
        }
    )

    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


def test_monotonicity_evaluation(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
):
    """
    Test the monotonicity metric on CIFAR-10 data.
    """
    metrics = monotonicity_cifar10(data_shape, ref_model, test_dataset)

    assert len(metrics) == 1
    monotonicity_metric: Measure = metrics[0]

    assert monotonicity_metric.name == "monotonicity"
    assert isinstance(monotonicity_metric.score, float)
    # monotonicity correlation can be between -1 and 1
    assert -1 <= monotonicity_metric.score <= 1
    assert isinstance(monotonicity_metric.time, datetime.datetime)


def test_monotonicity_value_evaluation(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
):
    """
    Test that monotonicity metric produces reasonable values.
    """
    metrics = monotonicity_cifar10(data_shape, ref_model, test_dataset)

    monotonicity_metric: Measure = metrics[0]

    # we expect positive monotonicity correlation for a well-trained model
    assert monotonicity_metric.score > 0, (
        f"Expected positive monotonicity for well-trained model, "
        f"got {monotonicity_metric.score}"
    )

    # the score should be reasonably high
    assert monotonicity_metric.score > 0.3, (
        f"Expected monotonicity > 0.3 for pretrained model, "
        f"got {monotonicity_metric.score}"
    )


# Evaluator functions
@pytest.mark.parametrize("evaluator_function", model_metric_registry)
def test_monotonicity_cifar10_single(
    evaluator_function: tuple[str, ModelMetric],
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
):
    measures = monotonicity_cifar10(data_shape, ref_model, test_dataset)
    save_measures(evaluator_function[0], measures)
    assert len(measures) > 0


@pytest.mark.parametrize("evaluator_function", model_metric_registry)
def test_monotonicity_cifar10_batched(
    evaluator_function: tuple[str, ModelMetric],
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
):
    original_data = test_dataset.data
    measures = []

    for i in range(0, len(original_data), 2):  # small batch size for test subset
        batch_end = min(i + 2, len(original_data))
        test_dataset.data = original_data.iloc[i:batch_end].copy()

        batch_measures = monotonicity_cifar10(data_shape, ref_model, test_dataset)
        measures.extend(batch_measures)

    test_dataset.data = original_data
    save_measures(evaluator_function[0], measures)

    assert len(measures) > 0
