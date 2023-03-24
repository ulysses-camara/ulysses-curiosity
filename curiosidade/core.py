"""Main entry point for probing models."""
import typing as t
import warnings
import collections

import torch
import torch.nn
import tqdm.auto

from . import helpers
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

        self.base_model = adapters.extensors.InferencePrunerExtensor(adapters.base.DummyAdapter())
        self.task: probers.tasks.base.BaseProbingTask = probers.tasks.base.DummyProbingTask()
        self.probers: dict[str, probers.ProbingModelWrapper] = {}
        self.is_trained = False

    def __repr__(self) -> str:
        pieces: list[str] = [f"{self.__class__.__name__}:"]

        model_str = str(self.base_model).replace("\n", "\n | ")

        pieces.append(f"(a): Base model: {model_str}")

        pieces.append(" |")

        pieces.append(f"(b): Task name: {self.task.task_name}")
        pieces.append("(c): Probing dataset(s):")

        def format_dl_info(split_name: str, dataloader: probers.tasks.base.DataLoaderType) -> str:
            split_name = f"({split_name})"
            return (
                f" | {split_name:<7}: {len(dataloader):4} batches of size (at most) "
                f"{dataloader.batch_size}."
            )

        pieces.append(format_dl_info("train", self.task.probing_dataloader_train))

        if self.task.has_eval:
            pieces.append(format_dl_info("eval", self.task.probing_dataloader_eval))  # type: ignore

        if self.task.has_test:
            pieces.append(format_dl_info("test", self.task.probing_dataloader_test))  # type: ignore

        if self.probers:
            pieces.append(f"(d): Probed module(s) ({len(self.probers)} in total):")

            for i, probed_modules_name in enumerate(self.probers.keys()):
                pieces.append(f" | ({i}): {probed_modules_name}")

        else:
            pieces.append("(d): No attached probing models.")

        return "\n".join(pieces)

    @property
    def task_name(self) -> str:
        """Return task name."""
        return self.task.task_name

    @property
    def probed_modules(self) -> tuple[str, ...]:
        """Return names of all probed modules."""
        return tuple(self.probers.keys())

    @property
    def pruned_modules(self) -> tuple[str, ...]:
        """Return names of all pruned modules."""
        if not isinstance(self.base_model, adapters.extensors.InferencePrunerExtensor):
            return tuple()  # type: ignore

        return self.base_model.pruned_module_names

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
        match_modules_to_attach_as_regex: bool = True,
        modules_input_dim: input_handlers.ModuleInputDimType = None,
        prune_unrelated_modules: t.Optional[t.Union[t.Sequence[str], t.Literal["infer"]]] = None,
        enable_cuda_in_inspection: bool = True,
    ) -> "ProbingModelContainer":
        """Attach probing models to specified `base_model` modules.

        Parameters
        ----------
        base_model : torch.nn.Module or transformers.PreTrainedModel
            Pretrained base model to attach probing models to.

        probing_model_factory : probers.ProbingModelFactory
            Probing model factory object.

        modules_to_attach : t.Pattern[str] or str or t.Sequence[str]
            A list or regular expression pattern specifying which model modules should be probed.
            Use `base_model.named_modules()` to check available model modules for probing.

        match_modules_to_attach_as_regex : bool, default=True
            If True, interpret `modules_to_attach` value as a regular expression in the form
            `^\\s*(?:modules_to_attach)\\s*$`. This argument only has effect when
            `type(modules_to_attach)=str`, otherwise it is ignored.

        modules_input_dim : dict[str, int] or None, default=None
            Input dimension of each probing model.

            - If mapping (dict), should map the module name to its corresponding input dimension.
              Input dimension of modules not present in this mapping will be inferred;
            - If None, the input dimensions will be inferred from the output dimensions sequences
              in `base_model.named_modules()`.

        prune_unrelated_modules : t.Sequence[str] or 'infer' or None, default=None
            Whether or not to prune pretrained modules unrelated to any probing model. This avoids
            unnecessary computations, speeding up the training procedure substantially if no probing
            model depends on final pretrained modules. Attempting to forward through pruned modules
            will interrupt immediately the forward flow, therefore saving computations since no
            further activations are required to train the probing models.

            - If 'infer', attempt to find the first module such that no probing model depends on
              any further module outputs. This heuristics only works properly if the model forward
              flow is deterministic and 'one-dimensional' (no bifurcations). This strategy is
              expected to work most of the time for any regular pretrained model;
            - If list, must contain the module names to prune;
            - If None, no module will be pruned, and the pretrained forward flow will be computed on
              its entirety.

        enable_cuda_in_inspection : bool, default=True
            Whether cuda can be enabled during probing input dimension inference, or while in search
            for unrelated modules in pretrained model. Argument used only if container device is set
            to 'cuda', otherwise it is ignored.

        Returns
        -------
        self
        """
        base_model = base_model.to("cpu")
        adapted_base_model = adapters.get_model_adapter(base_model)
        self.base_model = adapters.extensors.InferencePrunerExtensor(adapted_base_model)

        self.task = probing_model_factory.task
        self.probers = {}
        inspection_result: dict[str, t.Any] = {}
        pruned_modules: dict[str, torch.nn.Module] = {}
        modules_input_dim = modules_input_dim or {}

        fn_module_is_probed = input_handlers.get_fn_select_modules_to_probe(
            modules_to_attach=modules_to_attach,
            literal_string_input=not match_modules_to_attach_as_regex,
        )

        probed_modules = {
            module_name: module
            for module_name, module in base_model.named_modules()
            if fn_module_is_probed(module_name)
        }

        if enable_cuda_in_inspection:
            self.base_model.to(self.device)

        inspection_result = helpers.run_inspection_batches(
            sample_batches=[next(iter(self.task.probing_dataloader_train))],
            base_model=self.base_model,
            probed_modules=probed_modules.keys(),
            known_output_dims=modules_input_dim.keys(),
        )

        unnecessary_modules = inspection_result["unnecessary_modules"]
        probing_input_dims = inspection_result["probing_input_dims"]
        probing_input_dims.update(modules_input_dim)

        self.base_model.to("cpu")

        for module_name, module in probed_modules.items():
            try:
                self.probers[module_name] = probing_model_factory.create_and_attach(
                    module=module,
                    probing_input_dim=probing_input_dims[module_name],
                    random_seed=self.random_seed,
                )

            except Exception as err:
                raise RuntimeError(
                    f"Could not attach prober in module '{module_name}' "
                    f"(input_dim: {probing_input_dims[module_name]})."
                ) from err

        if prune_unrelated_modules == "infer":
            pruned_modules = dict(unnecessary_modules)

        elif prune_unrelated_modules:
            pruned_modules = input_handlers.find_modules_to_prune(
                module_names_to_prune=prune_unrelated_modules,
                named_modules=tuple(self.base_model.named_modules()),
                probed_module_names=frozenset(self.probers.keys()),
            )

        self.base_model.register_pruned_modules(pruned_modules)

        if not self.probers:
            warnings.warn(
                message=(
                    "No probing module has been attached. One probable cause is format mismatch of "
                    "values in the parameter 'modules_to_attach' and base model's named weights. "
                    + (
                        f"Also {match_modules_to_attach_as_regex=}. Is it the expected behaviour? "
                        if not match_modules_to_attach_as_regex
                        else ""
                    )
                ),
                category=UserWarning,
            )

        self._check_container_length_match(
            current_container=pruned_modules.keys(),
            expected_container=prune_unrelated_modules,
            message="Some of modules to prune were not found in pretrained model",
        )

        self._check_container_length_match(
            current_container=self.probers.keys(),
            expected_container=modules_to_attach,
            message="Some of the provided modules were not effectively attached",
        )

        return self

    def detach(self) -> "ProbingModelContainer":
        """Detach and remove all attached probers, if any."""
        for _, item in self.probers.items():
            item.detach()

        self.probers.clear()
        return self

    @staticmethod
    def _check_container_length_match(
        current_container: t.Collection[t.Any], expected_container: t.Any, message: str
    ) -> None:
        """Issue a warning if containers length does not match."""
        expected_is_container = not isinstance(expected_container, str) and hasattr(
            expected_container, "__len__"
        )

        if not expected_is_container or len(current_container) == len(expected_container):
            return

        unmatched_expected_items: list[str] = sorted(
            set(expected_container) - set(current_container)
        )
        unmatched_expected_items = list(map(lambda item: f" - {item}", unmatched_expected_items))
        warnings.warn(
            message=(
                f"{message}:\n"
                + "\n".join(unmatched_expected_items)
                + "\nThis may be due format mismatch of module name and the provided names."
            ),
            category=UserWarning,
        )

    def _run_epoch_train(
        self,
        dataloader: probers.tasks.base.DataLoaderType,
        gradient_accumulation_steps: int = 1,
        show_progress_bar: bool = False,
        mv_avg_loss_momentum: float = 0.95,
    ) -> output_handlers.MetricPack:
        """Run a full training epoch."""
        self.base_model.eval()

        res = output_handlers.MetricPack()
        pbar = tqdm.auto.tqdm(dataloader, disable=not show_progress_bar)

        for prober in self.probers.values():
            prober.train()

        mv_avg_loss = -1.0
        min_loss = float("+inf")
        max_loss = float("-inf")

        for i, batch in enumerate(pbar, 1):
            with torch.no_grad():
                batch_input_feats, batch_input_labels = self.base_model.break_batch(batch)
                self.base_model(batch_input_feats)

            batch_input_labels = batch_input_labels.to(self.device)
            accumulate_grad = (i % gradient_accumulation_steps != 0) and i < len(dataloader)

            with torch.set_grad_enabled(True):
                cur_avg_loss = 0.0

                for module_name, prober in self.probers.items():
                    metrics = prober.step(
                        input_labels=batch_input_labels,
                        accumulate_grad=accumulate_grad,
                        is_test=False,
                    )

                    res.append(metrics, module_name, i)
                    cur_avg_loss += metrics["loss"]

                cur_avg_loss /= len(self.probers)

                if mv_avg_loss < 0.0:
                    mv_avg_loss = cur_avg_loss
                else:
                    mv_avg_loss = (mv_avg_loss - cur_avg_loss) * mv_avg_loss_momentum + cur_avg_loss

                min_loss = min(min_loss, cur_avg_loss)
                max_loss = max(max_loss, cur_avg_loss)

                pbar.set_description(
                    f"train: {min_loss=:8.6f} {max_loss=:8.6f} {mv_avg_loss=:8.6f}"
                )

        return res

    def _run_epoch_eval(
        self,
        dataloader: probers.tasks.base.DataLoaderType,
        show_progress_bar: bool = False,
        mv_avg_loss_momentum: float = 0.95,
    ) -> output_handlers.MetricPack:
        """Run a full validation epoch."""
        self.base_model.eval()

        prober_outputs: t.Dict[str, t.List[t.Tuple[torch.Tensor, torch.Tensor]]]
        prober_outputs = collections.defaultdict(list)

        res = output_handlers.MetricPack()
        pbar = tqdm.auto.tqdm(dataloader, disable=not show_progress_bar)

        for prober in self.probers.values():
            prober.eval()

        mv_avg_loss = -1.0
        min_loss = float("+inf")
        max_loss = float("-inf")

        with torch.no_grad():
            for batch in pbar:
                batch_input_feats, batch_input_labels = self.base_model.break_batch(batch)
                self.base_model(batch_input_feats)

                batch_input_labels = batch_input_labels.to(self.device)

                cur_avg_loss = 0.0

                for module_name, prober in self.probers.items():
                    metrics = prober.step(
                        input_labels=batch_input_labels,
                        is_test=True,
                        compute_metrics=False,
                    )

                    cur_avg_loss += metrics["loss"]

                    if prober.task.has_metrics:
                        prober_outputs[module_name].append(
                            (
                                prober.output_tensor.cpu(),
                                batch_input_labels.cpu(),
                            )
                        )

                cur_avg_loss /= len(self.probers)

                if mv_avg_loss < 0.0:
                    mv_avg_loss = cur_avg_loss
                else:
                    mv_avg_loss = (mv_avg_loss - cur_avg_loss) * mv_avg_loss_momentum + cur_avg_loss

                min_loss = min(min_loss, cur_avg_loss)
                max_loss = max(max_loss, cur_avg_loss)

                pbar.set_description(f"test: {min_loss=:8.6f} {max_loss=:8.6f} {mv_avg_loss=:8.6f}")

            for module_name, prober in self.probers.items():
                if module_name not in prober_outputs or not prober.task.has_metrics:
                    continue

                output_tensor = torch.cat([out for out, _ in prober_outputs[module_name]], dim=0)
                input_labels = torch.cat([lab for _, lab in prober_outputs[module_name]], dim=0)

                output_tensor = output_tensor.to(self.device)
                input_labels = input_labels.to(self.device)

                loss_val = float(
                    prober.task.loss_fn(output_tensor, input_labels).detach().cpu().item()
                )
                metrics_fn_out = prober.task.metrics_fn(output_tensor, input_labels)  # type: ignore

                metrics_fn_out["loss"] = loss_val
                res.append(metrics_fn_out, module_name, -1)

        return res

    def train(
        self,
        num_epochs: int = 1,
        gradient_accumulation_steps: int = 1,
        show_progress_bar: t.Literal["epoch", True, None] = None,
    ) -> output_handlers.ProbingResults:
        """Train probing models.

        Parameters
        ----------
        num_epochs : int, default=1
            Number of training epochs.

        gradient_accumulation_steps : int, default=1
            Number of batches before one weight update.

        show_progress_bar : {'epoch', True, None}, default=None

            - If 'epoch', display a progress bar for each epoch.
            - If True, display a single progress bar for the entire training procedure.
            - If None, progress bar is omitted.

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

        if show_progress_bar not in {True, None, "epoch"}:
            raise ValueError(
                f"'show_progress_bar' must be True, None, or 'epoch' (got '{show_progress_bar}')."
            )

        self.base_model.to(self.device)
        for prober in self.probers.values():
            prober.to(self.device)

        metrics_train = output_handlers.MetricPack()
        metrics_eval = output_handlers.MetricPack()
        metrics_test = output_handlers.MetricPack()

        self.is_trained = True

        pbar = tqdm.auto.tqdm(range(num_epochs), disable=show_progress_bar is not True)
        show_progress_bar_per_epoch = show_progress_bar == "epoch"

        with torch.random.fork_rng(enabled=self.random_seed is not None):
            if self.random_seed is not None:
                torch.random.manual_seed(self.random_seed)

            for epoch in pbar:
                metrics_train.combine(
                    self._run_epoch_train(
                        dataloader=self.task.probing_dataloader_train,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        show_progress_bar=show_progress_bar_per_epoch,
                    ).expand_key_dim(epoch)
                )

                if self.task.has_eval:
                    metrics_eval.combine(
                        self._run_epoch_eval(
                            dataloader=self.task.probing_dataloader_eval,  # type: ignore
                        ).expand_key_dim(epoch)
                    )

                for prober in self.probers.values():
                    if prober.has_lr_scheduler:
                        prober.step_lr_scheduler()

            if self.task.has_test:
                metrics_test.combine(
                    self._run_epoch_eval(
                        dataloader=self.task.probing_dataloader_test,  # type: ignore
                        show_progress_bar=show_progress_bar_per_epoch,
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
    match_modules_to_attach_as_regex: bool = True,
    modules_input_dim: input_handlers.ModuleInputDimType = None,
    prune_unrelated_modules: t.Optional[t.Union[t.Sequence[str], t.Literal["infer"]]] = None,
    device: t.Union[torch.device, str] = "cpu",
    random_seed: t.Optional[int] = None,
) -> ProbingModelContainer:
    """Attach probing models to specified `base_model` modules.

    Parameters
    ----------
    base_model : torch.nn.Module
        Pretrained base model to attach probing models to.

    probing_model_factory : probers.ProbingModelFactory
        Probing model factory object.

    modules_to_attach : t.Pattern[str] or str or t.Sequence[str]
        A list or regular expression pattern specifying which model modules should be probed.
        Use `base_model.named_modules()` to check available model modules for probing.

    match_modules_to_attach_as_regex : bool, default=True
        If True, interpret `modules_to_attach` value as a regular expression in the form
        `^\\s*(?:modules_to_attach)\\s*$`. This argument only has effect when `modules_to_attach`
        type is `str`, otherwise it is ignored.

    modules_input_dim : dict[str, int] or None, default=None
        Input dimension of each probing model.

        - If mapping (dict), should map the module name to its corresponding input dimension.
          Input dimension of modules not present in this mapping will be inferred.
        - If None, the input dimensions will be inferred from the output dimensions sequences
          in `base_model.named_modules()`.

    prune_unrelated_modules : t.Sequence[str] or 'infer' or None, default=None
        Whether or not to prune pretrained modules unrelated to any probing model. This avoids
        unnecessary computations, speeding up the training procedure substantially if no probing
        model depends on final pretrained modules. Attempting to forward through pruned modules
        will interrupt immediately the forward flow, therefore saving computations since no
        further activations are required to train the probing models.

        - If 'infer', will attemp to find the first module such that no probing model depends on
          any further module outputs. This heuristics only works properly if the model forward
          flow is deterministic and 'one-dimensional' (no bifurcations). This strategy is expected
          to work most of the time for any regular pretrained model.
        - If list, must contain the module names to prune.
        - If None, no module will be pruned, and the pretrained forward flow will be computed on
          its entirety.

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
        match_modules_to_attach_as_regex=match_modules_to_attach_as_regex,
        modules_input_dim=modules_input_dim,
        prune_unrelated_modules=prune_unrelated_modules,
    )

    return prober_container
