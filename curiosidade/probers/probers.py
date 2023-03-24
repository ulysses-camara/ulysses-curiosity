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

    lr_scheduler : torch.optim.lr_scheduler._LRScheduler or None, default=None
        Optional learning rate scheduler, coupled with `optim`.
    """

    def __init__(
        self,
        probing_model: torch.nn.Module,
        task: tasks.base.BaseProbingTask,
        optim: torch.optim.Optimizer,
        lr_scheduler: t.Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.probing_model = probing_model
        self.task = task

        self.input_tensors: t.Union[dict[str, t.Any], tuple[torch.Tensor, ...]]
        self.input_tensors = (torch.empty(0, dtype=torch.float64),)

        self.output_tensor = torch.empty(0, dtype=torch.float64)
        self.loss = torch.empty(0)

        self.input_source_hook: t.Optional[torch.utils.hooks.RemovableHandle] = None
        self.attached_module: t.Optional[torch.nn.Module] = None

    def __repr__(self) -> str:
        pieces: list[str] = [f"{self.__class__.__name__}"]

        pieces.append("  (a): Probing model:")
        pieces.append("    " + str(self.probing_model).replace("\n", "\n    "))

        pieces.append(f"  (b): Optimizer: {self.optim.__class__.__name__}")
        pieces.append(
            "  (c): Learning rate scheduler: "
            f"{self.lr_scheduler.__class__.__name__ if self.lr_scheduler else None}"
        )
        pieces.append(f"  (d): Task: {self.task.task_name}")
        pieces.append(f"  (e): Is attached: {self.is_attached}")

        return "\n".join(pieces)

    @property
    def is_attached(self) -> bool:
        """Check whether the probing model is attached to a pretrained module."""
        return self.attached_module is not None

    @property
    def has_lr_scheduler(self) -> bool:
        """Check whether the probing model has a learning rate scheduler."""
        return self.lr_scheduler is not None

    def attach(self, module: torch.nn.Module) -> "ProbingModelWrapper":
        """Attach probing model to `module`."""
        self.detach()

        def fn_hook_forward(
            layer: torch.nn.Module,
            l_input: torch.Tensor,
            l_output: t.Union[tuple[torch.Tensor, ...], torch.Tensor, dict[str, t.Any]],
        ) -> None:
            # pylint: disable='unused-argument'
            if torch.is_tensor(l_output):
                self.input_tensors = (l_output.detach(),)  # type: ignore

            elif issubclass(type(l_output), dict):
                self.input_tensors = {
                    key: item.detach() if torch.is_tensor(item) else item
                    for key, item in l_output.items()  # type: ignore
                }

            else:
                self.input_tensors = tuple(
                    tensor.detach()  # type: ignore
                    for tensor in l_output
                    if torch.is_tensor(tensor)
                )

        self.attached_module = module
        self.input_source_hook = module.register_forward_hook(fn_hook_forward)

        return self

    def detach(self) -> "ProbingModelWrapper":
        """Detach attached prober, if any."""
        if self.attached_module is not None:
            self.attached_module = None

        if self.input_source_hook is not None:
            self.input_source_hook.remove()
            self.input_source_hook = None

        return self

    def remove(self) -> "ProbingModelWrapper":
        """Alias for 'detach'."""
        return self.detach()

    def to(self, device: t.Union[torch.device, str]) -> "ProbingModelWrapper":
        """Move probing model to `device`."""
        # pylint: disable='invalid-name'
        self.probing_model.to(device)
        return self

    def step(
        self,
        input_labels: torch.Tensor,
        accumulate_grad: bool = False,
        is_test: bool = False,
        compute_metrics: bool = True,
    ) -> dict[str, float]:
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

        compute_metrics : bool, default=True
            If True, compute metrics related to the task for the current batch.

        Returns
        -------
        metrics : dict[str float]
            Metrics related to the current batch.
        """
        should_backward = not is_test
        should_optim_step = not accumulate_grad and not is_test

        if isinstance(self.input_tensors, dict):
            self.output_tensor = self.probing_model(**self.input_tensors)

        else:
            self.output_tensor = self.probing_model(*self.input_tensors)

        self.loss = self.task.loss_fn(self.output_tensor, input_labels)

        if should_backward:
            self.loss.backward()

        if should_optim_step:
            self.optim.step()
            self.optim.zero_grad()

        self.output_tensor = self.output_tensor.detach()
        loss_val = float(self.loss.detach().cpu().item())

        metrics = {"loss": loss_val}

        if compute_metrics and self.task.has_metrics:
            metrics_fn_out = self.task.metrics_fn(self.output_tensor, input_labels)  # type:ignore

            try:
                metrics.update(metrics_fn_out)

            except TypeError as err:
                raise TypeError(
                    "Provided 'metrics_fn' function must return a dictionary in the format "
                    f"{{metric_name: metric_value}} (got type '{type(metrics_fn_out) = }')."
                ) from err

        return metrics

    def step_lr_scheduler(self, *args: t.Any, **kwargs: t.Any) -> "ProbingModelWrapper":
        """Apply one step of learning rate scheduler."""
        self.lr_scheduler.step(*args, **kwargs)  # type: ignore
        return self

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
    probing_model_fn : t.Callable[..., torch.nn.Module]
        Probing model factory function, or class derived from torch.nn.Module. Must receive its
        input dimension (an integer) as its first positional argument, and its output dimension
        (also an integer) as its second positional argument. Extra arguments can be handled via
        `extra_kwargs` parameter.

    task : task.base.BaseProbingTask
        Probing task related to the probing models.

    optim_fn : t.Type[torch.optim.Optimizer], default=torch.optim.Adam
        Optimizer factory function.

    lr_scheduler_fn : t.Type[torch.optim.lr_scheduler._LRScheduler], default=None
        If provided, will set up a learning rate scheduler coupled to the optimizer.

    extra_kwargs : dict[str, t.Any] or None, default=None
        Extra arguments to provide to `probing_model_fn`.

    Examples
    --------
    >>> import curiosidade
    >>> import functools
    ...
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
    >>> task = curiosidade.probers.base.DummyProbingTask()
    >>> ProbingModelFactory(
    ...     probing_model_fn=ProbingModel,  # Note: do not instantiate.
    ...     optim_fn=functools.partial(torch.optim.Adam, lr=0.01),  # Note: do not instantiate.
    ...     task=task,
    ... )
    ProbingModelFactory
      (a): probing model generator : <class 'curiosidade.probers.probers.ProbingModel'>
      (b): optimizer generator : functools.partial(<class 'torch.optim.adam.Adam'>, lr=0.01)
      (c): task : 'unnamed_task' (classification)
    """

    def __init__(
        self,
        probing_model_fn: t.Callable[..., torch.nn.Module],
        task: tasks.base.BaseProbingTask,
        optim_fn: t.Callable[
            [t.Iterator[torch.nn.Parameter]], torch.optim.Optimizer
        ] = torch.optim.Adam,
        lr_scheduler_fn: t.Optional[t.Type[torch.optim.lr_scheduler._LRScheduler]] = None,
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
        self.lr_scheduler_fn = lr_scheduler_fn
        self.extra_kwargs = extra_kwargs or {}

    def __repr__(self) -> str:
        pieces: list[str] = [f"{self.__class__.__name__}"]

        pieces.append(f"  (a): probing model generator : {self.probing_model_fn}")
        pieces.append(f"  (b): optimizer generator : {self.optim_fn}")
        pieces.append(f"  (c): task : '{self.task.task_name}' ({self.task.task_type})")

        return "\n".join(pieces)

    def create_and_attach(
        self,
        module: torch.nn.Module,
        probing_input_dim: t.Union[int, tuple[int, ...]],
        random_seed: t.Optional[int] = None,
    ) -> ProbingModelWrapper:
        """Create a brand-new probing model and attach it to `module`.

        Parameters
        ----------
        module : torch.nn.Module
            Module to attach the probing model.

        probing_input_dim : int or tuple[int, ...]
            Input dimension of probing model. It should match the output dimension of `module`.
            If `module` has more than one output, must be a tuple with the dimensions of each
            output, mantaining the order. This tuple will be unpacked before provided to the
            probing model.

        random_seed : int or None, default=None
            Random seed set while creating the probing model, mainly to control for random weight
            initialization, and any other non-deterministic behaviours. Note that this only take
            into account Torch-related pseudo-randomness. If your model depends on other independent
            pseudo-random generators (such as `random` or `numpy.random`), you must control their
            behaviour separately within the probing model code (for instance, providing a random
            seed via `ProbingModelFactory.extra_kwargs`).

        Returns
        -------
        probing_model : ProbingModelWrapper
            Probing model created and attached to `module`.
        """
        probing_output_dim = self.task.output_dim

        if not hasattr(probing_input_dim, "__len__"):
            probing_input_dim = (probing_input_dim,)  # type: ignore

        with torch.random.fork_rng(enabled=random_seed is not None):
            if random_seed is not None:
                torch.random.manual_seed(random_seed)

            probing_model = self.probing_model_fn(
                *probing_input_dim,  # type: ignore
                probing_output_dim,
                **self.extra_kwargs,
            )

        optim = self.optim_fn(probing_model.parameters())
        lr_scheduler = self.lr_scheduler_fn(optim) if self.lr_scheduler_fn else None

        probing_module = ProbingModelWrapper(
            probing_model=probing_model,
            optim=optim,
            lr_scheduler=lr_scheduler,
            task=self.task,
        )

        probing_module.attach(module)

        return probing_module

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> ProbingModelWrapper:
        """Call `create_and_attach`."""
        return self.create_and_attach(*args, **kwargs)
