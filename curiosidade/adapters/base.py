"""Base class for model adapters."""
import typing as t
import abc

import torch


# pylint: disable='invalid-name'


class BaseAdapter(abc.ABC):
    """Base class for model adapters.

    Model adapters are used to provide an unified API from distinct base model
    formats (such as PyTorch modules and Huggingface transformers).
    """

    def __init__(self, model: torch.nn.Module, device: t.Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self._pruned_modules: dict[str, torch.nn.Module] = {}

    def __repr__(self) -> str:
        return str(self.model)

    @property
    def pruned_modules_names(self):
        return tuple(self._pruned_modules.keys())

    @property
    def pruned_modules(self):
        return tuple(self._pruned_modules.values())

    @property
    def has_pruned_modules(self) -> bool:
        return len(self._pruned_modules) > 0

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
    def named_modules(self) -> t.Iterator[tuple[str, torch.nn.Module]]:
        """Return Torch module .named_modules() iterator."""

    @staticmethod
    @abc.abstractmethod
    def break_batch(batch: t.Any) -> tuple[t.Any, torch.Tensor]:
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
        super().__init__(model=torch.nn.Identity(), device="cpu")

    @staticmethod
    def break_batch(batch: t.Any) -> tuple[t.Any, torch.Tensor]:
        """Dummy no-op."""
        return None, torch.empty(0)

    def forward(self, X: t.Any) -> t.Any:
        """Dummy no-op."""
        return None

    def named_modules(self) -> t.Iterator[tuple[str, torch.nn.Module]]:
        """Dummy no-op."""
        return iter(tuple())
