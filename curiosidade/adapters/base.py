"""Base class for model adapters."""
import typing as t
import abc

import torch


AdapterInferenceOutputType = tuple[t.Any, t.Any, torch.Tensor]


class BaseAdapter(abc.ABC):
    """Base class for model adapters.

    Model adapters are used to provide an unified API from distinct base model
    formats (such as PyTorch modules and Huggingface transformers).
    """

    def __init__(self, model: t.Any, device: t.Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)

    def __repr__(self) -> str:
        return str(self.model)

    def to(self, device: t.Union[str, torch.device]) -> "BaseAdapter":
        """Move model to `device`."""
        # pylint: disable='invalid-name'
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def eval(self) -> "BaseAdapter":
        """Set model to eval model."""
        self.model.eval()
        return self

    def train(self) -> "BaseAdapter":
        """Set model to train model."""
        self.model.train()
        return self

    @abc.abstractmethod
    def forward(self, batch: t.Any) -> AdapterInferenceOutputType:
        """Model forward pass.

        Returns
        -------
        out : t.Any
            Forward pass output.

        X : t.Any
            Input features (batch without `labels`).

        y : torch.Tensor
            Label features.
        """

    def __call__(
        self, *args: t.Any, **kwargs: t.Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(*args, **kwargs)


class DummyAdapter(BaseAdapter):
    """Dummy adapter used as placeholder."""

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        # pylint: disable='unused-argument'
        dummy = torch.Parameter(torch.empty(0), requires_grad=False)
        super().__init__(model=dummy, device="cpu")

    def forward(self, batch: t.Any) -> AdapterInferenceOutputType:
        return None, None, torch.empty(0)
