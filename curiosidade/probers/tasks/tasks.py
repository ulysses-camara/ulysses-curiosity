"""Probing task classes."""
from . import base


class ProbingTaskSentenceLength(base.BaseProbingTask):
    pass


class ProbingTaskWordContent(base.BaseProbingTask):
    pass


class ProbingTaskBigramShift(base.BaseProbingTask):
    pass


class ProbingTaskTreeDepth(base.BaseProbingTask):
    pass


class ProbingTaskTopConstituent(base.BaseProbingTask):
    pass


class ProbingTaskTense(base.BaseProbingTask):
    pass


class ProbingTaskSubjectNumber(base.BaseProbingTask):
    pass


class ProbingTaskObjectNumber(base.BaseProbingTask):
    pass


class ProbingTaskSOMO(base.BaseProbingTask):
    pass


class ProbingTaskCoordinationInversion(base.BaseProbingTask):
    pass


class ProbingTaskCustom(base.BaseProbingTask):
    """Custom probing task.

    Parameters
    ----------
    probing_dataloader_train : torch.utils.data.DataLoader
        Train probing dataloader.

    probing_dataloader_eval : torch.utils.data.DataLoader or None, default=None
        Evaluation probing dataloader.

    probing_dataloader_test : torch.utils.data.DataLoader or None, default=None
        Test probing dataloader.

    loss_fn : t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function related to the probing task.

    task_name : str, default="unnamed_task"
        Probing task name.
    """
