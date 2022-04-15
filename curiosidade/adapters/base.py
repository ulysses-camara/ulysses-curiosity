import typing as t
import abc

import torch


AdapterInferenceOutputType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class BaseAdapter(abc.ABC):
    def __init__(self, model: t.Any, device: t.Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)

    def __repr__(self):
        return str(self.model)

    def to(self, device: t.Union[str, torch.device]) -> "BaseAdapter":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def eval(self) -> "BaseAdapter":
        self.model.eval()
        return self

    def train(self) -> "BaseAdapter":
        self.model.train()
        return self

    @abc.abstractmethod
    def forward(self, batch: t.Any) -> AdapterInferenceOutputType:
        pass

    def __call__(
        self, *args: t.Any, **kwargs: t.Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(*args, **kwargs)
