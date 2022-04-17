"""Main entry point for probing models."""
import typing as t
import warnings

import torch
import torch.nn
import tqdm.auto

from . import probers
from . import adapters
from . import input_handlers
from . import output_handlers


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

        self.base_model: adapters.base.BaseAdapter = adapters.base.DummyAdapter()
        self.task: probers.tasks.base.BaseProbingTask = probers.tasks.base.DummyProbingTask()
        self.probers: dict[str, probers.ProbingModelWrapper] = {}
        self.is_trained = False

    def __repr__(self) -> str:
        pieces: list[str] = [f"{self.__class__.__name__}:"]

        pieces.append(f"(a): Base model: {self.base_model}")
        pieces.append(f"(b): Task name: {self.task.task_name}")
        pieces.append("(c): Probing datasets:")

        def format_dl_info(split_name: str, dataloader: torch.utils.data.DataLoader) -> str:
            split_name = f"({split_name})"
            return (
                f"  {split_name:<7}: {len(dataloader):4} batches of size (at most) "
                f"{dataloader.batch_size}."
            )

        pieces.append(format_dl_info("train", self.task.probing_dataloader_train))

        if self.task.has_eval:
            pieces.append(format_dl_info("eval", self.task.probing_dataloader_eval))

        if self.task.has_test:
            pieces.append(format_dl_info("test", self.task.probing_dataloader_test))

        if self.probers:
            pieces.append(f"(d): Probed modules ({len(self.probers)} in total):")

            for i, key in enumerate(self.probers.keys()):
                pieces.append(f"  ({i}): {key}")

        else:
            pieces.append(f"(d): No attached probing models.")

        return "\n".join(pieces)

    @property
    def task_name(self) -> str:
        """Return task name."""
        return self.task.task_name

    @property
    def probed_modules(self) -> tuple[str, ...]:
        """Return names of all probed modules."""
        return tuple(self.probers.keys())

    def __getitem__(self, key: str) -> probers.ProbingModelWrapper:
        return self.probers[key]

    def __iter__(self) -> t.Iterator[tuple[str, probers.ProbingModelWrapper]]:
        return iter(self.probers.items())

    def __len__(self) -> int:
        return len(self.probers)

    def attach(
        self,
        base_model: torch.nn.Module,
        probing_model_factory: probers.ProbingModelFactory,
        modules_to_attach: t.Union[t.Pattern[str], str, t.Sequence[str]],
        modules_input_dim: input_handlers.ModuleInputDimType = None,
    ) -> "ProbingModelContainer":
        """Attach probing models to specificied `base_model` modules.

        Parameters
        ----------
        base_model : torch.nn.Module or transformers.PreTrainedModel
            Pretrained base model to attach probing models to.

        probing_model_factory : probers.ProbingModelFactory
            Probing model factory object.

        modules_to_attach : t.Pattern[str] or str or t.Sequence[str]
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
        self.probers = {}

        fn_module_is_probed = input_handlers.get_fn_select_modules_to_probe(modules_to_attach)

        prev_output_dim: int = 0

        for module_name, module in base_model.named_modules():
            if hasattr(module, "out_features"):
                prev_output_dim = module.out_features

            if not fn_module_is_probed(module_name):
                continue

            module_output_dim = input_handlers.get_probing_model_input_dim(
                modules_input_dim=modules_input_dim,
                default_dim=prev_output_dim,
                module_name=module_name,
                module_index=len(self.probers),
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
                    "values in the parameter 'modules_to_attach' and base model's named weights."
                ),
                category=UserWarning,
            )

        is_container = not isinstance(modules_to_attach, str) and hasattr(
            modules_to_attach, "__len__"
        )

        if is_container and count_attached_modules < len(modules_to_attach):
            probed_modules = set(self.probers.keys())
            not_attached_modules: list[str] = sorted(set(modules_to_attach) - probed_modules)
            not_attached_modules = list(map(lambda item: f" - {item}", not_attached_modules))
            warnings.warn(
                message=(
                    "Some of the provided modules were not effectively attached:\n"
                    + "\n".join(not_attached_modules)
                    + "\nThis may be due format mismatch of module name and the provided names."
                ),
                category=UserWarning,
            )

        return self

    def _run_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        gradient_accumulation_steps: int = 1,
        is_test: bool = False,
        show_progress_bar: bool = False,
    ) -> output_handlers.MetricPack:
        """Run a full training epoch."""
        self.base_model.eval()

        res = output_handlers.MetricPack()
        pbar = tqdm.auto.tqdm(dataloader, disable=not show_progress_bar)

        for prober in self.probers.values():
            if is_test:
                prober.eval()
            else:
                prober.train()

        for i, batch in enumerate(pbar, 1):
            with torch.no_grad():
                *_, batch_input_labels = self.base_model(batch=batch)

            batch_input_labels = batch_input_labels.to(self.device)
            accumulate_grad = i % gradient_accumulation_steps != 0

            with torch.set_grad_enabled(not is_test):
                for module_name, prober in self.probers.items():
                    metrics = prober.step(
                        input_labels=batch_input_labels,
                        accumulate_grad=accumulate_grad,
                        is_test=is_test,
                    )
                    res.append(metrics, module_name, i)

        return res

    def train(
        self,
        num_epochs: int = 1,
        gradient_accumulation_steps: int = 1,
        show_progress_bar: bool = False,
    ) -> output_handlers.ProbingResults:
        """Train probing models.

        Parameters
        ----------
        num_epochs : int, default=1
            Number of training epochs.

        gradient_accumulation_steps : int, default=1
            Number of batches before one weight update.

        show_progress_bar : bool, default=False
            If True, display a progress bar for each epoch.

        Returns
        -------
        results : output_handlers.ProbingResults
            Probing results, separated per split (train, eval, and test).
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

        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "'gradient_accumulation_steps' must be >= 1 "
                f"(got {gradient_accumulation_steps = })."
            )

        self.base_model.to(self.device)
        for prober in self.probers.values():
            prober.to(self.device)

        metrics_train = output_handlers.MetricPack()
        metrics_eval = output_handlers.MetricPack()
        metrics_test = output_handlers.MetricPack()

        self.is_trained = True

        with torch.random.fork_rng(enabled=self.random_seed is not None):
            if self.random_seed is not None:
                torch.random.manual_seed(self.random_seed)

            for epoch in range(num_epochs):
                metrics_train.combine(
                    self._run_epoch(
                        dataloader=self.task.probing_dataloader_train,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        show_progress_bar=show_progress_bar,
                    ).expand_key_dim(epoch)
                )

                if self.task.has_eval:
                    metrics_eval.combine(
                        self._run_epoch(
                            dataloader=self.task.probing_dataloader_eval,
                            is_test=True,
                        ).expand_key_dim(epoch)
                    )

            if self.task.has_test:
                metrics_test.combine(
                    self._run_epoch(
                        dataloader=self.task.probing_dataloader_test,
                        is_test=True,
                    ).expand_key_dim(-1)
                )

        self.base_model.to("cpu")
        for prober in self.probers.values():
            prober.to("cpu")

        ret = output_handlers.ProbingResults(
            train=metrics_train,
            eval=metrics_eval or None,
            test=metrics_test or None,
        )

        return ret


def attach_probers(
    base_model: torch.nn.Module,
    probing_model_factory: probers.ProbingModelFactory,
    modules_to_attach: t.Union[t.Pattern[str], str, t.Sequence[str]],
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

    modules_to_attach : t.Pattern[str] or str or t.Sequence[str]
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
