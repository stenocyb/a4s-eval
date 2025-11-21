from datetime import datetime
import traceback
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, Saliency, GradientShap

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.data_model.measure import Measure
from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.service.model_functional import FunctionalModel

import quantus

def _load_cifar10_model() -> nn.Module:
    """
    Loads a PyTorch ResNet-18 model structure (CIFAR-10).
    """
    
    try:
        model = models.resnet18(weights=None)
        
        # modify for CIFAR-10
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        
        # load pretrained CIFAR-10 checkpoint
        checkpoint_url = "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
        model.load_state_dict(state_dict)
                
    except Exception:
        traceback.print_exc()
        return None

    model.eval()
    return model


@model_metric(name="monotonicity")
def monotonicity_cifar10(
    datashape: DataShape, # unused; keep for compatibility
    model: Model,
    dataset: Dataset,
    functional_model: FunctionalModel,
) -> list[Measure]:
    """
    Measures if increasing feature attribution scores leads to increasing model confidence,
    returning the result as a list of Measure objects.
    """
    explanation_method = "integrated_gradients"
    num_samples = 3
    
    # setup data transformations and load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
    ])
    
    try:
        test_set = datasets.CIFAR10(root='./tests/data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Error loading CIFAR-10 dataset: {e}")
        return [Measure(name="monotonicity", score=0.0, time=datetime.now())]
        
    # load the specific PyTorch model
    torch_model = _load_cifar10_model()
    if torch_model is None:
        return [Measure(name="monotonicity", score=0.0, time=datetime.now())]
        
    num_samples = min(num_samples, len(test_set))
    
    # extract & prepare data samples
    test_indices = list(range(num_samples))
    subset = torch.utils.data.Subset(test_set, test_indices)
    
    data_loader = torch.utils.data.DataLoader(subset, batch_size=num_samples)
    try:
        x_tensor, y_labels_tensor = next(iter(data_loader))
    except StopIteration:
        print("Failed to load any samples.")
        return [Measure(name="monotonicity", score=0.0, time=datetime.now())]

    y_labels = y_labels_tensor.cpu().numpy()
    
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
        with torch.no_grad():
            outputs = torch_model(x_tensor)
            predicted_classes = torch.argmax(outputs, dim=1)
            target_tensor = predicted_classes
        
        if explanation_method == "gradient_shap":
            baseline = torch.zeros_like(x_tensor)
            attr_batch = explainer.attribute(x_tensor, baselines=baseline, target=target_tensor, n_steps=10)
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