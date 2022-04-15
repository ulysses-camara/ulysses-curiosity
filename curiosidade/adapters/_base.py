import typing as t
import abc

import torch


class BaseAdapter(abc.ABC):
    @abc.abstractmethod
    def forward(
        self, batch: t.Any, model: t.Any, device: t.Union[str, torch.device]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def __call__(
        self, *args: t.Any, **kwargs: t.Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(*args, **kwargs)
