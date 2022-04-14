"""Main entry point for probers and tasks."""
import typing as t
import collections
import warnings

import torch
import torch.nn
import tqdm.auto

from . import probers
from . import tasks


class Probers:
    """Collection of probing models.

    Parameters
    ----------
    task : tasks.base.BaseTask
        A Task object.

    optim_fn : t.Type[torch.optim.Optimizer]
        A torch.optim factory (callable).

    device : {'cpu', 'cuda'}
        Device to train probing models.
    """

    def __init__(
        self,
        task: tasks.base.BaseTask,
        optim_fn: t.Type[torch.optim.Optimizer] = torch.optim.Adam,
        device: t.Union[torch.device, str] = "cpu",
    ):
        if not hasattr(optim_fn, "__call__"):
            raise TypeError(
                "Expected a callable (factory) in 'optim_fn' parameter, but received "
                f"'{type(optim_fn)}'. Please make sure to provide a optimizer type, not "
                "an instantiated optimizer. If you need to custom any optimizer parameter, "
                "you can provide it by using 'functools.partial(optim_fn, param1=value1, "
                "param2=value2, ...)'."
            )

        self.base_model = torch.nn.Module()
        self.task = task
        self.optim_fn = optim_fn
        self.probers: dict[str, probers.ProbingModule] = dict()
        self.device = torch.device(device)
        self.is_trained = False

    def __repr__(self):
        pieces: list[str] = ["ProberPack:"]

        pieces.append(f"(a): Base model: {self.base_model}")
        pieces.append(f"(b): Task name: {self.task.task_name}")

        pieces.append(f"(c): Probed modules ({len(self.probers)} in total):")

        for i, key in enumerate(self.probers.keys()):
            pieces.append(f"  ({i}): {key}")

        return "\n".join(pieces)

    @property
    def task_name(self):
        return self.task.task_name

    def __iter__(self):
        return iter(self.probers)

    def __len__(self):
        return len(self.probers)

    def attach(
        self,
        base_model: torch.nn.Module,
        probing_model_fn: t.Callable[[int, ...], torch.nn.Module],
        layers_to_attach: t.Sequence[str],
        probing_model_kwargs: t.Optional[dict[str, t.Any]] = None,
    ):
        self.base_model = base_model.to("cpu")
        self.probers = dict()

        layers_to_attach = frozenset(layers_to_attach)

        probing_model_kwargs = probing_model_kwargs or {}
        probing_input_dim: int = 0
        count_attached_modules: int = 0

        for param_key, param_val in self.base_model.named_modules():
            if hasattr(param_val, "out_features"):
                probing_input_dim = param_val.out_features

            if param_key not in layers_to_attach:
                continue

            module_probing_model = probing_model_fn(probing_input_dim, **probing_model_kwargs)
            module_probing_model = module_probing_model.to("cpu")

            self.probers[param_key] = probers.ProbingModule(
                probing_model=module_probing_model,
                task=self.task,
                optim_fn=self.optim_fn,
                source_layer=param_val,
            )

            count_attached_modules += 1

        if count_attached_modules == 0:
            base_model_named_modules = [name for name, _ in self.base_model.named_modules() if name]

            warnings.warn(
                message=(
                    "No probing modules were attached. One probable cause is format mismatch of "
                    f"values in the parameter 'layers_to_attach' ({', '.join(layers_to_attach)}) "
                    f"and base model's named weights ({', '.join(base_model_named_modules)})."
                ),
                category=UserWarning,
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

    @staticmethod
    def _flatten_results(res: dict[t.Any, dict[t.Any, list[t.Any]]]) -> dict[t.Any, list[t.Any]]:
        flattened_res = collections.defaultdict(list)

        for _, val_dict in res.items():
            for key, vals in val_dict.items():
                flattened_res[key].extend(vals)

        return flattened_res

    def train(
        self,
        num_epochs: int = 1,
        show_progress_bar: bool = False,
        flatten_results: bool = True,
    ) -> dict[int, dict[str, list[float]]]:
        if self.is_trained:
            warnings.warn(
                message=(
                    "Probing weights are already pretrained from previous run. If this is not "
                    "intended, attach probing models again."
                ),
                category=UserWarning,
            )

        if not self.probers:
            raise RuntimeError(
                "No probing models were attached to base model. Please call "
                f"'{self.__class__.__name__}.attach(...)' "
                f"(or '{__name__}.transformer_model(...)') "
                "first before training the probing models."
            )

        self.base_model.to(self.device)
        for probers in self.probers.values():
            probers.to(self.device)

        res: dict[int, dict[str, list[float]]] = {}

        for epoch in range(num_epochs):
            res[epoch] = self._run_epoch(show_progress_bar=show_progress_bar)

        self.base_model.to("cpu")
        for probers in self.probers.values():
            probers.to("cpu")

        if flatten_results:
            res = self._flatten_results(res)

        self.is_trained = True

        return res


def attach_probers(
    base_model: torch.nn.Module,
    probing_model_fn: torch.nn.Module,
    task,
    layers_to_attach: t.Sequence[str],
    optim_fn: t.Type[torch.optim.Optimizer] = torch.optim.Adam,
    device: t.Union[torch.device, str] = "cpu",
    probing_model_kwargs: t.Optional[dict[str, t.Any]] = None,
) -> Probers:
    """

    Parameters
    ----------

    Returns
    -------

    See Also
    --------
    """
    prober_pack = Probers(optim_fn=optim_fn, task=task, device=device)

    prober_pack.attach(
        base_model=base_model,
        probing_model_fn=probing_model_fn,
        layers_to_attach=layers_to_attach,
        probing_model_kwargs=probing_model_kwargs,
    )

    return prober_pack
