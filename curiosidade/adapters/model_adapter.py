import typing as t

import transformers
import torch
import torch.nn

from . import _base


class HuggingfaceAdapter(_base.BaseAdapter):
    def forward(
        self, batch: dict[str, torch.Tensor], model: t.Any, device: t.Union[str, torch.device]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = batch.pop("labels")

        for key, val in batch.items():
            batch[key] = val.to(device)

        X = batch
        out = model(**X)

        return out, X, y


class TorchModuleAdapter(_base.BaseAdapter):
    def forward(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, ...],
        model: t.Any,
        device: t.Union[str, torch.device],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X, y, *_ = batch
        X = X.to(device)
        out = model(X)
        return out, X, y


def get_model_adapter(model: t.Any) -> _base.BaseAdapter:
    if isinstance(model, transformers.PreTrainedModel):
        return HuggingfaceAdapter()

    if isinstance(model, torch.nn.Module):
        return TorchModuleAdapter()

    raise TypeError(
        f"Unknown model type '{type(model)}'. Please provide a Huggingface transformer or "
        "a PyTorch module."
    )
