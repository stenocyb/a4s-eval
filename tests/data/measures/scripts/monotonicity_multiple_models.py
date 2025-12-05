import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from captum.attr import IntegratedGradients
import quantus
import pandas as pd
import os
from typing import Dict, List, Any

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # use GPU if available for performance reasons
print(f"Running on device: {DEVICE}")


def build_pretrained_cifar_models() -> Dict[str, nn.Module]:
    """
    Load pretrained CIFAR-10 models
    """
    model_zoo = {}

    # we use the chenyaofo/pytorch-cifar-models repo from Torch Hub
    # provide ResNets and VGGs that expect 32x32 input
    repo = "chenyaofo/pytorch-cifar-models"

    print("Downloading/Loading pre-trained models from Torch Hub...")

    # ResNet-20, CIFAR variant of ResNet, roughly similar depth to 18 but lighter
    model_zoo["resnet20"] = torch.hub.load(repo, "cifar10_resnet20", pretrained=True)

    # ResNet-32, deeper variant
    model_zoo["resnet32"] = torch.hub.load(repo, "cifar10_resnet32", pretrained=True)

    # VGG-11, batch normalized
    model_zoo["vgg11_bn"] = torch.hub.load(repo, "cifar10_vgg11_bn", pretrained=True)

    for name, model in model_zoo.items():
        model.to(DEVICE)
        model.eval()

    return model_zoo


def load_cifar10_batch(num_samples: int = 10):
    """
    Standard CIFAR-10 normalization.
    Returns a batch of images [10, 3, 32, 32]
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_set = datasets.CIFAR10(
        root="./a4s-eval/tests/data", train=False, download=True, transform=transform
    )

    loader = torch.utils.data.DataLoader(
        test_set, batch_size=num_samples, shuffle=False
    )
    x, y = next(iter(loader))

    return x.to(DEVICE), y.to(DEVICE)


def compute_attributions(model: nn.Module, x_batch: torch.Tensor):
    """
    Computes attributions for the whole batch at once
    """
    with torch.no_grad():
        preds = model(x_batch)
    targets = torch.argmax(preds, dim=1)

    explainer = IntegratedGradients(model)

    attr = explainer.attribute(
        x_batch, target=targets, n_steps=20, internal_batch_size=10
    )

    return attr.detach().cpu().numpy(), targets.cpu()


def evaluate_monotonicity(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    attr_batch: np.ndarray,
) -> List[float]:
    """
    Quantus metric loop
    """
    scores = []
    metric = quantus.MonotonicityCorrelation(
        nr_samples=10,
        abs=True,
        normalise=True,
        disable_warnings=True,
    )

    for i in range(len(x_batch)):
        # slicing single sample
        x_i = x_batch[i : i + 1].cpu().numpy()
        y_i = y_batch[i : i + 1].cpu().numpy()
        a_i = attr_batch[i : i + 1]

        # calculate score
        raw_score = metric(
            model=model,
            x_batch=x_i,
            y_batch=y_i,
            a_batch=a_i,
            explain_func=None,
            device=DEVICE,
        )

        if isinstance(raw_score, list):
            scores.append(float(raw_score[0]))
        else:
            scores.append(float(raw_score))

    return scores


def save_measures(
    results: Dict[str, Dict[str, Any]], metric_name: str, filename: str
) -> None:
    """
    Dumps the results into a CSV file. (model name, mean & individual scores)
    """
    data_list = []

    # each model gets a row
    for model_name, data in results.items():
        row_dict = {
            "model": model_name,
            "metric": metric_name,
            "mean_score": data["mean"],
        }

        for i, score in enumerate(data["scores"]):
            row_dict[f"score_{i}"] = score

        data_list.append(row_dict)

    df = pd.DataFrame(data_list)

    output_path = os.path.join("..", filename)
    df.to_csv(output_path, index=False)


def main():
    print("Loading CIFAR-10 data...")
    x_batch, y_batch = load_cifar10_batch(num_samples=10)
    models_dict = build_pretrained_cifar_models()

    print(f"\nProcessing {len(models_dict)} models on {len(x_batch)} images each...")

    results = {}
    metric_name = "monotonicity"
    csv_filename = "monotonicity_different_models.csv"

    for name, model in models_dict.items():
        print(f"\n--- {name} ---")

        print("> Generating attributions...")
        attributions, _ = compute_attributions(model, x_batch)

        print("> Calculating Monotonicity...")
        scores = evaluate_monotonicity(model, x_batch, y_batch, attributions)

        mean_score = np.mean(scores)

        print(f"- Scores: {[round(s, 4) for s in scores]}")
        print(f"- Mean:   {mean_score:.4f}")

        results[name] = {
            "mean": mean_score,
            "scores": scores,
            "metric": metric_name,
        }

    save_measures(results=results, metric_name=metric_name, filename=csv_filename)
    print(f"Saved results to {csv_filename}.")


if __name__ == "__main__":
    main()
