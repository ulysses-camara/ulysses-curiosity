"""Probing model wrappers."""
import typing as t

import torch
import torch.nn

from . import tasks


class ProbingModelWrapper:
    """Probing model wrapper.

    Parameters
    ----------
    probing_model : torch.nn.Module
        Probing model.

    task : task.base.BaseProbingTask
        Probing task related to the probing model.

    optim : torch.optim.Optimizer
        Optimizer used to train `probing_model` parameters.
    """

    def __init__(
        self,
        probing_model: torch.nn.Module,
        task: tasks.base.BaseProbingTask,
        optim: torch.optim.Optimizer,
    ):
        self.input_tensor = torch.empty(0, dtype=torch.float64)
        self.output_tensor = torch.empty(0, dtype=torch.float64)
        self.input_source_hook: t.Optional[torch.utils.hooks.RemovableHandle] = None
        self.optim = optim
        self.probing_model = probing_model
        self.task = task
        self.loss = torch.empty(0)

    def attach(self, module: torch.nn.Module) -> "ProbingModelWrapper":
        """Attach probing model to `module`."""

        def fn_hook_forward(
            layer: torch.nn.Module, l_input: torch.Tensor, l_output: torch.Tensor
        ) -> None:
            # pylint: disable='unused-argument'
            self.input_tensor = l_output.detach()

        self.input_source_hook = module.register_forward_hook(fn_hook_forward)

        return self

    def to(self, device: t.Union[torch.device, str]) -> "ProbingModelWrapper":
        """Move probing model to `device`."""
        # pylint: disable='invalid-name'
        self.probing_model.to(device)
        return self

    def step(
        self, input_labels: torch.Tensor, accumulate_grad: bool = False, is_test: bool = False
    ) -> float:
        """Perform a single optimization step with `input_labels` as target reference.

        Parameters
        ----------
        input_labels : torch.Tensor
            Ground truth labels for current batch.

        accumulate_grad : bool, default=False
            If True, will not perform gradient cleaning, adding the current backward computation
            to the pre-existing gradient. This also prevent updates to model weights.

        is_test : bool, default=False
            If True, does not compute backward gradients is this run. Also prevent weight
            update, gradient accumulation, and gradient cleaning.

        Returns
        -------
        loss : float
            Loss value related to the current batch.
        """
        should_backward = not is_test
        should_optim_step = not accumulate_grad and not is_test

        self.output_tensor = self.probing_model(self.input_tensor)
        self.loss = self.task.loss_fn(self.output_tensor, input_labels)

        if should_backward:
            self.loss.backward()

        if should_optim_step:
            self.optim.step()
            self.optim.zero_grad()

        loss_val = float(self.loss.cpu().detach().item())

        return loss_val

    def train(self) -> "ProbingModelWrapper":
        """Set model to train mode."""
        self.probing_model.train()
        return self

    def eval(self) -> "ProbingModelWrapper":
        """Set model to evaluation mode."""
        self.probing_model.eval()
        return self


class ProbingModelFactory:
    """Factory to create multiple probing models from a single configuration.

    Parameters
    ----------
    probing_model_fn : t.Callable[[int], torch.nn.Module]
        Probing model factory function, or class derived from torch.nn.Module. Must receive its
        input dimension (an integer) as its first positional argument, and its output dimension
        (also an integer) as its second positional argument. Extra arguments can be handled via
        `extra_kwargs` parameter.

    task : task.base.BaseProbingTask
        Probing task related to the probing models.

    optim_fn : t.Type[torch.optim.Optimizer], default=torch.optim.Adam
        Optimizer factory function.

    extra_kwargs : dict[str, t.Any] or None, default=None
        Extra arguments to provide to `probing_model_fn`.

    Examples
    --------
    >>> import functools
    >>> class ProbingModel(torch.nn.Module):
    ...     def __init__(self, input_dim: int, output_dim: int):
    ...         super().__init__()
    ...         self.params = torch.nn.Sequential(
    ...             torch.nn.Linear(input_dim, 20),
    ...             torch.nn.ReLU(inplace=True),
    ...             torch.nn.Linear(20, output_dim),
    ...         )
    ...
    ...     def forward(self, X):
    ...         return self.params(X)
    ...
    >>> task = curiosidade.ProbingTaskSentenceLength()
    >>> ProbingModelFactory(
    ...     probing_model_fn=ProbingModel,  # Note: do not instantiate.
    ...     optim_fn=functools.partial(torch.optim.Adam, lr=0.01),  # Note: do not instantiate.
    ...     task=task,
    ... )
    ProbingModelFactory
      (a): probing model generator : <class 'ProbingModel'>
      (b): optimizer generator : functools.partial(<class 'torch.optim.adam.Adam'>, lr=0.01)
      (c): task : 'sentence length (sentlen)' (classification)
    """

    def __init__(
        self,
        probing_model_fn: t.Callable[[int], torch.nn.Module],
        task: tasks.base.BaseProbingTask,
        optim_fn: t.Type[torch.optim.Optimizer] = torch.optim.Adam,
        extra_kwargs: t.Optional[dict[str, t.Any]] = None,
    ):
        if not hasattr(optim_fn, "__call__"):
            raise TypeError(
                "Expected a callable (factory) in 'optim_fn' parameter, but received "
                f"'{type(optim_fn)}'. Please make sure to provide a optimizer type, not "
                "an instantiated optimizer. If you need to custom any optimizer parameter, "
                "you can provide it by using 'functools.partial(optim_fn, param1=value1, "
                "param2=value2, ...)'."
            )

        self.probing_model_fn = probing_model_fn
        self.task = task
        self.optim_fn = optim_fn
        self.extra_kwargs = extra_kwargs or {}

    def __repr__(self) -> str:
        pieces: list[str] = [f"{self.__class__.__name__}"]

        pieces.append(f"  (a): probing model generator : {self.probing_model_fn}")
        pieces.append(f"  (b): optimizer generator : {self.optim_fn}")
        pieces.append(f"  (c): task : '{self.task.task_name}' ({self.task.task_type})")

        return "\n".join(pieces)

    def create_and_attach(
        self, module: torch.nn.Module, probing_input_dim: int, random_seed: t.Optional[int] = None
    ) -> ProbingModelWrapper:
        """Create a brand-new probing model and attach it to `module`.

        Parameters
        ----------
        module : torch.nn.Module
            Module to attach the probing model.

        probing_input_dim : int
            Input dimension of probing model. It should match the output dimension of `module`.

        random_seed : int or None, default=None
            Random seed to instantiate the probing model, controlling for random weight
            initialization, and any other non-deterministic behaviours.
        """
        probing_output_dim = self.task.output_dim

        with torch.random.fork_rng(enabled=random_seed is not None):
            if random_seed is not None:
                torch.random.manual_seed(random_seed)

            probing_model = self.probing_model_fn(
                probing_input_dim,
                probing_output_dim,
                **self.extra_kwargs,
            )

        optim = self.optim_fn(probing_model.parameters())

        probing_module = ProbingModelWrapper(
            probing_model=probing_model,
            optim=optim,
            task=self.task,
        )

        probing_module.attach(module)

        return probing_module

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> ProbingModelWrapper:
        """Call `create_and_attach`."""
        return self.create_and_attach(*args, **kwargs)
