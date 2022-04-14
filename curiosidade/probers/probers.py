import typing as t
import functools

import torch
import torch.nn

from . import _base


class ProbingModule(_base.BaseProber):
    def __init__(
        self,
        probing_model: torch.nn.Module,
        task,
        optim_fn: t.Type[torch.optim.Optimizer],
        source_layer: torch.nn.Module,
    ):
        super().__init__(
            probing_model=probing_model,
            task=task,
            optim_fn=optim_fn,
        )

        self.input_tensor = torch.empty(0, dtype=torch.float64)
        self.output_tensor = torch.empty(0, dtype=torch.float64)

        def fn_hook_forward(
            layer: torch.nn.Module, l_input: torch.Tensor, l_output: torch.Tensor
        ) -> None:
            self.input_tensor = l_output.detach()

        self.input_source_hook = source_layer.register_forward_hook(fn_hook_forward)

    def to(self, device: t.Union[torch.device, str]) -> "ProbingModule":
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
