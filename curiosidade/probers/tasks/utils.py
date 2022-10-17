import typing as t

import torch


__all__ = [
    "get_standard_metrics_fn",
]


MetricFnType = t.Callable[[torch.Tensor, torch.Tensor], dict[str, float]]


def get_standard_metrics_fn(num_classes: int) -> MetricFnType:
    try:
        import torchmetrics

    except ImportError as err:
        raise ImportError(
            "Package 'torchmetrics' not found. This package is necessary for a preconfigured "
            "'metrics_fn' of this probing task. Please install it, or provide your own "
            "'metrics_fn'."
        ) from err

    accuracy_fn = torchmetrics.Accuracy(num_classes=num_classes)
    f1_fn = torchmetrics.F1Score(num_classes=num_classes)

    def metrics_fn(logits: torch.Tensor, truth_labels: torch.Tensor) -> dict[str, float]:
        accuracy = accuracy_fn(logits, truth_labels).detach().cpu().item()
        f1 = f1_fn(logits, truth_labels).detach().cpu().item()
        return {"accuracy": accuracy, "f1": f1}

    return metrics_fn
