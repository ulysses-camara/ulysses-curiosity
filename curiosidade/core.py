"""Main entry point for probing models."""
import typing as t
import collections
import warnings

import regex
import torch
import torch.nn
import tqdm.auto

from . import probers
from . import adapters
from . import input_handlers

try:
    import transformers

    BaseModelType = t.Union[torch.nn.Module, transformers.PreTrainedModel]

except ImportError:
    BaseModelType = torch.nn.Module


class ProbingModelContainer:
    """Container of probing models.

    Parameters
    ----------
    device : {'cpu', 'cuda'}, default='cpu'
        Device type to train probing models.

    random_seed : int or None, default=None
        If specified, this random seed will be used while instantiating the probing models and
        also during their training, ensuring reprodutible results.
    """

    def __init__(
        self,
        device: t.Union[torch.device, str] = "cpu",
        random_seed: t.Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.random_seed = random_seed

        self.base_model: adapters.base.BaseAdapter = None
        self.probers: dict[str, probers.ProbingModule] = dict()
        self.is_trained = False

    def __repr__(self):
        pieces: list[str] = [f"{self.__class__.__name__}:"]

        pieces.append(f"(a): Base model: {self.base_model}")
        pieces.append(f"(b): Task name: {self.task.task_name}")
        pieces.append(
            f"(c): Probing dataset size: {len(self.task.probing_dataloader)} batches of "
            f"size (at most) {self.task.probing_dataloader.batch_size}."
        )

        pieces.append(f"(d): Probed modules ({len(self.probers)} in total):")

        for i, key in enumerate(self.probers.keys()):
            pieces.append(f"  ({i}): {key}")

        return "\n".join(pieces)

    @property
    def task_name(self) -> str:
        """Return task name."""
        return self.task.task_name

    @property
    def probed_modules(self) -> tuple[str, ...]:
        return tuple(self.probers.keys())

    def __iter__(self):
        return iter(self.probers)

    def __len__(self):
        return len(self.probers)

    def attach(
        self,
        base_model: BaseModelType,
        probing_model_factory: probers.ProbingModelFactory,
        modules_to_attach: t.Union[regex.Pattern, str, t.Sequence[str]],
        modules_input_dim: input_handlers.ModuleInputDimType = None,
    ) -> "ProbingModelContainer":
        """Attach probing models to specificied `base_model` modules.

        Parameters
        ----------
        base_model : torch.nn.Module or transformers.PreTrainedModel
            Pretrained base model to attach probing models to.

        probing_model_factory : probers.ProbingModelFactory
            Probing model factory object.

        modules_to_attach : regex.Pattern or str or t.Sequence[str]
            A list or regular expression pattern specifying which model modules should be probed.
            Use `base_model.named_modules()` to check available model modules for probing.

        modules_input_dim : t.Sequence[int] or dict[str, int] or None, default=None
            Input dimension of each probing model.
            - If list, the dimension in the i-th index should correspond to the input dimension of
              the i-th probing model.
            - If mapping (dict), should map the module name to its corresponding input dimension.
              Input dimension of modules not present in this mapping will be inferred.
            - If None, the input dimensions will be inferred from the output dimensions sequences
              in `base_model.named_modules()`.

        Returns
        -------
        self
        """
        base_model = base_model.to("cpu")

        self.base_model = adapters.get_model_adapter(base_model)
        self.task = probing_model_factory.task
        self.probers = dict()

        fn_module_is_probed = input_handlers.get_fn_select_modules_to_probe(modules_to_attach)

        prev_output_dim: int = 0

        for module_name, module in base_model.named_modules():
            if hasattr(module, "out_features"):
                prev_output_dim = module.out_features

            if not fn_module_is_probed(module_name):
                continue

            module_output_dim = input_handlers.get_probing_model_input_dim(
                modules_output_dim=modules_output_dim,
                default_dim=prev_output_dim,
                module_name=module_name,
                module_index=count_attached_modules,
            )

            self.probers[module_name] = probing_model_factory.create_and_attach(
                module=module,
                probing_input_dim=module_output_dim,
                random_seed=self.random_seed,
            )

        count_attached_modules = len(self.probers)

        if count_attached_modules == 0:
            warnings.warn(
                message=(
                    "No probing modules were attached. One probable cause is format mismatch of "
                    f"values in the parameter 'modules_to_attach' and base model's named weights."
                ),
                category=UserWarning,
            )

        is_container = not isinstance(modules_to_attach, str) and hasattr(
            modules_to_attach, "__len__"
        )

        if is_container and count_attached_modules < len(modules_to_attach):
            probed_modules = set(self.probers.keys())
            not_attached_modules = sorted(set(modules_to_attach) - probed_modules)
            not_attached_modules = map(lambda item: f" - {item}", not_attached_modules)
            warnings.warn(
                message=(
                    "Some of the provided modules were not effectively attached:\n"
                    + "\n".join(not_attached_modules)
                    + "\nThis may be due format mismatch of module name and the provided names."
                ),
                category=UserWarning,
            )

        return self

    def _run_epoch(self, show_progress_bar: bool = False) -> dict[str, list[float]]:
        """Run a full training epoch."""
        self.base_model.eval()

        res: dict[str, list[float]] = collections.defaultdict(list)

        for batch in tqdm.auto.tqdm(self.task.probing_dataloader, disable=not show_progress_bar):
            with torch.no_grad():
                *_, y = self.base_model(batch=batch)

            y = y.to(self.device)

            for prober_name, prober in self.probers.items():
                loss = prober.step(y)
                res[prober_name].append(loss)

        return res

    @staticmethod
    def _flatten_results(res: dict[t.Any, dict[t.Any, list[t.Any]]]) -> dict[t.Any, list[t.Any]]:
        """Merge results from all training epochs into a single list per module."""
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
    ) -> dict[t.Union[str, int], t.Union[list[float], dict[str, list[float]]]]:
        """Train probing models.

        Parameters
        ----------
        num_epochs : int, default=1
            Number of training epochs.

        show_progress_bar : bool, default=False
            If True, display a progress bar for each epoch.

        flatten_result : bool, default=True
            If True, results from every training epoch will be merged into a single list per
            probed module. If False, return results separed hierarchically; first by epoch, then
            by probed module.

        Returns
        -------
        train_results : dict
        """
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

        with torch.random.fork_rng(enabled=self.random_seed is not None):
            if self.random_seed is not None:
                torch.random.manual_seed(self.random_seed)

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
    probing_model_factory: probers.ProbingModelFactory,
    modules_to_attach: t.Union[regex.Pattern, str, t.Sequence[str]],
    modules_input_dim: input_handlers.ModuleInputDimType = None,
    device: t.Union[torch.device, str] = "cpu",
    random_seed: t.Optional[int] = None,
) -> ProbingModelContainer:
    """Attach probing models to specificied `base_model` modules.

    Parameters
    ----------
    base_model : torch.nn.Module
        Pretrained base model to attach probing models to.

    probing_model_factory : probers.ProbingModelFactory
        Probing model factory object.

    modules_to_attach : regex.Pattern or str or t.Sequence[str]
        A list or regular expression pattern specifying which model modules should be probed.
        Use `base_model.named_modules()` to check available model modules for probing.

    modules_input_dim : t.Sequence[int] or dict[str, int] or None, default=None
        Input dimension of each probing model.
        - If list, the dimension in the i-th index should correspond to the input dimension of
          the i-th probing model.
        - If mapping (dict), should map the module name to its corresponding input dimension.
          Input dimension of modules not present in this mapping will be inferred.
        - If None, the input dimensions will be inferred from the output dimensions sequences
          in `base_model.named_modules()`.

    device : {'cpu', 'cuda'}, default='cpu'
        Device used to train probing models.

    random_seed : int or None, default=None
        If specified, this random seed will be used while instantiating the probing models and
        also during their training, ensuring reprodutible results.

    Returns
    -------
    probers : ProbingModelContainer
        Container with every instantiated probing model, prepared for training.

    See Also
    --------
    ProbingModelContainer.train : train returned ProbingModelContainer instance.
    functools.partial : create a factory with preset named arguments.
    """
    prober_container = ProbingModelContainer(device=device, random_seed=random_seed)

    prober_container.attach(
        base_model=base_model,
        probing_model_factory=probing_model_factory,
        modules_to_attach=modules_to_attach,
        modules_input_dim=modules_input_dim,
    )

    return prober_container
