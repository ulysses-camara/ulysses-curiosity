import typing as t
import functools

import torch
import torch.nn

from . import tasks


class ProbingModel:
    def __init__(
        self,
        probing_model: torch.nn.Module,
        task: tasks.base.BaseProbingTask,
        optim: torch.optim.Optimizer,
    ):
        self.input_tensor = torch.empty(0, dtype=torch.float64)
        self.output_tensor = torch.empty(0, dtype=torch.float64)
        self.input_source_hook = None
        self.optim = optim
        self.probing_model = probing_model
        self.task = task

    def attach(self, module: torch.nn.Module) -> "ProbingModel":
        def fn_hook_forward(
            layer: torch.nn.Module, l_input: torch.Tensor, l_output: torch.Tensor
        ) -> None:
            self.input_tensor = l_output.detach()

        self.input_source_hook = module.register_forward_hook(fn_hook_forward)

        return self

    def to(self, device: t.Union[torch.device, str]) -> "ProbingModel":
        self.probing_model.to(device)
        return self

    def step(self, input_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.optim.zero_grad()
        self.output_tensor = self.probing_model(self.input_tensor)
        self.loss = self.task.loss_fn(input=self.output_tensor, target=input_labels)
        self.loss.backward()
        self.optim.step()

        loss_val = float(self.loss.cpu().detach().item())

        return loss_val


class ProbingModelFactory:
    def __init__(
        self,
        probing_model_fn: t.Callable[[int, ...], torch.nn.Module],
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

    def create_and_attach(self, module: torch.nn.Module, input_dim: int) -> ProbingModel:
        probing_model = self.probing_model_fn(input_dim, **self.extra_kwargs)
        optim = self.optim_fn(probing_model.parameters())

        probing_module = ProbingModel(
            probing_model=probing_model,
            optim=optim,
            task=self.task,
        )

        probing_module.attach(module)

        return probing_module

    def __call__(self, *args, **kwargs) -> ProbingModel:
        return self.create_and_attach(*args, **kwargs)
