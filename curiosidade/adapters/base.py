"""Base class for model adapters."""
import typing as t
import abc

import torch


class BaseAdapter(abc.ABC):
    """Base class for model adapters.

    Model adapters are used to provide an unified API from distinct base model
    formats (such as PyTorch modules and Huggingface transformers).
    """

    def __init__(
        self,
        model: t.Union[torch.nn.Module, "BaseAdapter"],
        device: t.Union[str, torch.device] = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)

    def __repr__(self) -> str:
        model_str = str(self.model).replace("\n", "\n |")
        return str(f"{self.__class__.__name__}({model_str})")

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

    @classmethod
    @abc.abstractmethod
    def break_batch(cls, batch: t.Any) -> tuple[t.Any, torch.Tensor]:
        """Break batch in inputs `input_feats` and input labels `input_labels` appropriately.

        Returns
        -------
        input_feats : t.Any
            Input features (batch without `labels`).

        input_labels : torch.Tensor
            Label features.
        """

    @abc.abstractmethod
    def forward(self, input_feats: t.Any) -> t.Any:
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

    @classmethod
    def break_batch(cls, batch: t.Any) -> tuple[t.Any, torch.Tensor]:
        """Dummy no-op."""
        return None, torch.empty(0)

    def forward(self, input_feats: t.Any) -> t.Any:
        """Dummy no-op."""
        return None

    def named_modules(self) -> t.Iterator[tuple[str, torch.nn.Module]]:
        """Dummy no-op."""
        return iter(tuple())


class BaseExtensor(BaseAdapter):
    """Base extensor to expand adapter capabilities."""

    def __init__(self, model: BaseAdapter):
        super().__init__(model=model, device=model.device)

    def named_modules(self) -> t.Iterator[tuple[str, torch.nn.Module]]:
        """Return Torch module .named_modules() iterator."""
        return self.model.named_modules()

    def break_batch(self, batch: t.Any) -> t.Any:
        """Break batch into proper input `input_feats` and labels `input_labels`."""
        # pylint: disable='arguments-differ'
        return self.model.break_batch(batch)

    def forward(self, input_feats: t.Any) -> t.Any:
        return self.model(input_feats)
