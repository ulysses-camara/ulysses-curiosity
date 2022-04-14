import typing as t
import copy
import collections

import torch
import torch.nn
import tqdm.auto

from . import probers
from . import tasks


class Probers:
    def __init__(self, task: tasks.base.BaseTask, optim_fn, device: str):
        self.base_model = torch.nn.Module()
        self.task = task
        self.optim_fn = optim_fn
        self.probers = dict()
        self.device = torch.device(device)

    def __repr__(self):
        pieces: list[str] = ["ProberPack:"]

        pieces.append(f"(a): Base model     : {self.base_model}")
        pieces.append(f"(b): Task           : {self.task.task_name}")

        pieces.append("(c): Probed modules :")

        for i, key in enumerate(self.probers.keys()):
            pieces.append(f"  ({i}): {key}")

        return "\n".join(pieces)

    def attach(
        self,
        base_model: torch.nn.Module,
        probing_model: torch.nn.Module,
        layers_to_attach: t.Sequence[str],
    ):
        self.base_model = base_model.to("cpu")
        self.probers: dict[str, probers.ProbingModule] = dict()

        for param_key, param_val in self.base_model.named_modules():
            if param_key not in layers_to_attach:
                continue

            self.probers[param_key] = probers.ProbingModule(
                probing_model=copy.deepcopy(probing_model.to("cpu")),
                task=self.task,
                optim_fn=self.optim_fn,
                source_layer=param_val,
            )

    def _run_epoch(self, show_progress_bar: bool = False) -> dict[str, list[float]]:
        self.base_model.eval()

        res: dict[str, list[float]] = collections.defaultdict(list)

        for X, y, *_ in tqdm.auto.tqdm(self.task.probing_dataloader, disable=not show_progress_bar):
            X = X.to(self.device)

            with torch.no_grad():
                self.base_model(X)

            y = y.to(self.device)

            for prober_name, prober in self.probers.items():
                loss = prober.step(y)
                res[prober_name].append(loss)

        return res

    def train(self, num_epochs: int = 1, show_progress_bar: bool = False) -> dict[int, dict[str, list[float]]]:
        self.base_model.to(self.device)
        for probers in self.probers.values():
            probers.to(self.device)

        res: dict[int, dict[str, list[float]]] = {}

        for epoch in range(num_epochs):
            res[epoch] = self._run_epoch(show_progress_bar=show_progress_bar)

        self.base_model.to("cpu")
        for probers in self.probers.values():
            probers.to("cpu")

        return res


def attach_probers(
    base_model: torch.nn.Module,
    probing_model: torch.nn.Module,
    optim_fn,
    task,
    layers_to_attach: t.Sequence[str],
    device: str = "cpu",
) -> Probers:
    prober_pack = Probers(optim_fn=optim_fn, task=task, device=device)

    prober_pack.attach(
        base_model=base_model,
        probing_model=probing_model,
        layers_to_attach=layers_to_attach,
    )
    return prober_pack
