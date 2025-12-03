# Metric Implementation Project

This repository contains my implementation of the Monotonicity metric into the A4S platform.

The repository can be found at `https://github.com/stenocyb/a4s-eval`

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/stenocyb/a4s-eval
   cd a4s-eval
   ```

2. Install dependencies using `uv`:

   ```bash
   uv sync
   ```

## Monotonicity metric

The metric is implemented in `a4s_eval/metrics/model_metrics/monotonicity_cifar10.py`

Tests are located in `tests/metrics/model_metrics/test_monotonicity_metric.py`

This metric measures if increasing feature attribution scores leads to increasing model confidence, specifically designed for CIFAR-10 image classification tasks.

### Running tests

```bash
uv run pytest tests/metrics/model_metrics/test_monotonicity_metric.py
```

The test suite includes:

- Basic functionality test (`test_monotonicity_evaluation`)
- Value range validation test (`test_monotonicity_value_evaluation`)
- Single evaluation test (`test_monotonicity_cifar10_single`)
- Batched evaluation test (`test_monotonicity_cifar10_batched`)

### Usage

```python
from a4s_eval.metrics.model_metrics.monotonicity_cifar10 import monotonicity_cifar10

measures = monotonicity_cifar10(
    datashape=data_shape,
    model=torch_model,
    dataset=dataset
)

monotonicity_score = measures[0].score
```

### Supported attribution methods

- `integrated_gradients` (default)
- `saliency`
- `gradient_shap`

### Test model

The test suite uses a pretrained ResNet-18 model on CIFAR-10 loaded from:

```
https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin
```

## Running predefined tests

To run all metric tests:

```bash
# Run monotonicity metric tests
uv run pytest tests/metrics/model_metrics/test_monotonicity_metric.py

# Run all tests
uv run pytest tests/
```
