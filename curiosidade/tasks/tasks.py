import torch

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
    def __init__(
        self,
        probing_dataloader: torch.utils.data.DataLoader,
        loss_fn,
        task_name: str = "unnamed_task",
    ):
        super().__init__(
            probing_dataloader=probing_dataloader,
            loss_fn=loss_fn,
            task_name=task_name,
        )
