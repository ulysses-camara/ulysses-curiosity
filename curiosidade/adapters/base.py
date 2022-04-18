"""Base class for model adapters."""
import typing as t
import abc

import torch


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

    def named_modules(self) -> t.Iterator[tuple[str, torch.nn.Module]]:
        """Return Torch module .named_modules() iterator."""
        return self.model.named_modules()

    @abc.abstractmethod
    def break_batch(self, batch: t.Any) -> tuple[t.Any, torch.Tensor]:
        """Break batch in inputs `X` and input labels `y` appropriately.

        Returns
        -------
        X : t.Any
            Input features (batch without `labels`).

        y : torch.Tensor
            Label features.
        """

    @abc.abstractmethod
    def forward(self, X: t.Any) -> t.Any:
        """Model forward pass.

        Returns
        -------
        out : t.Any
            Forward pass output.
        """

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self.forward(*args, **kwargs)


class DummyAdapter(BaseAdapter):
    """Dummy adapter used as placeholder."""

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        # pylint: disable='unused-argument'
        dummy = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        super().__init__(model=dummy, device="cpu")

    def break_batch(self, batch: t.Any) -> tuple[t.Any, torch.Tensor]:
        return None, torch.empty(0)

    def forward(self, X: t.Any) -> t.Any:
        return None
