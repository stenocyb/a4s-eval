from datetime import datetime
import numpy as np
import torch
from captum.attr import IntegratedGradients, Saliency, GradientShap

from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.data_model.measure import Measure
from a4s_eval.data_model.evaluation import DataShape, Dataset, Model

import quantus


@model_metric(name="monotonicity")
def monotonicity_cifar10(
    datashape: DataShape, model: Model, dataset: Dataset
) -> list[Measure]:
    """
    Checks if features ranked important by the explainer are also the ones that most change the model output when perturbed.

    Arguments:
        datashape: DataShape object containing feature information
        model:     The PyTorch model
        dataset:   Dataset object containing the test data

    Returns:
        List containing a single Measure object with the monotonicity score
    """
    explanation_method = "integrated_gradients"

    # extract model
    torch_model = model
    if torch_model is None:
        return [Measure(name="monotonicity", score=0.0, time=datetime.now())]

    torch_model.eval()

    # extract data from dataset, features & labels
    data_df = dataset.data

    try:
        if "image" in data_df.columns:
            x_tensor = torch.stack(list(data_df["image"].values))
        else:
            raise ValueError("Dataset must contain 'image' column with tensor data")

        if datashape.target and datashape.target.name in data_df.columns:
            y_labels = data_df[datashape.target.name].values
        else:
            raise ValueError("Dataset must contain target column")

    except Exception as e:
        print(f"Error extracting data: {e}")
        return [Measure(name="monotonicity", score=0.0, time=datetime.now())]

    # initialize attribution method
    if explanation_method == "integrated_gradients":
        explainer = IntegratedGradients(torch_model)
    elif explanation_method == "saliency":
        explainer = Saliency(torch_model)
    elif explanation_method == "gradient_shap":
        explainer = GradientShap(torch_model)
    else:
        raise ValueError(f"Unknown explanation method: {explanation_method}")

    # generate attributions
    try:
        outputs = torch_model(x_tensor)
        predicted_classes = torch.argmax(outputs, dim=1)
        target_tensor = predicted_classes

        if explanation_method == "gradient_shap":
            baseline = torch.zeros_like(x_tensor)
            attr_batch = explainer.attribute(
                x_tensor, baselines=baseline, target=target_tensor, n_steps=10
            )
        else:
            attr_batch = explainer.attribute(x_tensor, target=target_tensor, n_steps=10)

        attributions = attr_batch.detach().cpu().numpy()

    except Exception as e:
        print(f"Batch attribution error: {e}")
        return [Measure(name="monotonicity", score=0.0, time=datetime.now())]

    # compute monotonicity score with Quantus
    metric = quantus.MonotonicityCorrelation(
        nr_samples=1,
        abs=True,
        normalise=True,
        disable_warnings=True,
    )

    x_batch_numpy = x_tensor.numpy()

    scores = metric(
        model=torch_model,
        x_batch=x_batch_numpy,
        y_batch=y_labels,
        a_batch=attributions,
        explain_func=None,
        device="cpu",
    )

    return [
        Measure(
            name="monotonicity",
            score=float(np.mean(scores)),
            time=datetime.now(),
        )
    ]
