# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
import typing
from typing import Any, Dict, List, Iterable, Tuple, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.abstract import WrapperMetric

from monai.metrics import DiceMetric

from overrides import override



class DiceMetricWrapper(Metric):
    '''
    Create a Torchmetrics wrapper on MONAI DiceMetric
    '''
    def __init__(self, threshold=0.5):
        super().__init__()
        self.dice_metric = DiceMetric(
                include_background=True,
                reduction="mean_batch",
                ignore_empty=False,
            )
        self.threshold = threshold
    
    def update(self, preds: Tensor, target:Tensor) -> None:
        #preds need to be logits or prob values in [0,1]
        if not torch.all((preds >= 0) * (preds <= 1)):
            # preds is logits, convert with sigmoid
            preds = preds.sigmoid()

        preds = preds.gt(self.threshold)
        self.dice_metric(preds, target)

    def compute(self) -> Tensor:
        return self.dice_metric.aggregate()

    @override
    def reset(self):
        super().reset()
        self.dice_metric.reset()


'''
---------------------------------------------------------------------------------------------------------
Torchmetric wrappers adapted for use

1. In update() and forward() methods of MultitaskWrapper,
    MODIFY task_targets args so that it also accepts a single target Tensor besides dict of target Tensors

2. In _convert_output() method of ClasswiseWrapper,  
    > MODIFY condition (not self.prefix) --> (self._prefix is None)
    > ADD the mean of the metric across all classes in output
---------------------------------------------------------------------------------------------------------
'''

class MultitaskWrapper(WrapperMetric):
    """Wrapper class for computing different metrics on different tasks in the context of multitask learning.

    In multitask learning the different tasks requires different metrics to be evaluated. This wrapper allows
    for easy evaluation in such cases by supporting multiple predictions and targets through a dictionary.
    Note that only metrics where the signature of `update` follows the standard `preds, target` is supported.

    Args:
        task_metrics:
            Dictionary associating each task to a Metric or a MetricCollection. The keys of the dictionary represent the
            names of the tasks, and the values represent the metrics to use for each task.
        prefix:
            A string to append in front of the metric keys. If not provided, will default to an empty string.
        postfix:
            A string to append after the keys of the output dict. If not provided, will default to an empty string.

    .. note::
        The use pre prefix and postfix allows for easily creating task wrappers for training, validation and test.
        The arguments are only changing the output keys of the computed metrics and not the input keys. This means
        that a ``MultitaskWrapper`` initialized as ``MultitaskWrapper({"task": Metric()}, prefix="train_")`` will
        still expect the input to be a dictionary with the key "task", but the output will be a dictionary with the key
        "train_task".

    Raises:
        TypeError:
            If argument `task_metrics` is not an dictionary
        TypeError:
            If not all values in the `task_metrics` dictionary is instances of `Metric` or `MetricCollection`
        ValueError:
            If `prefix` is not a string
        ValueError:
            If `postfix` is not a string

    Example (with a single metric per class):
         >>> import torch
         >>> from torchmetrics.wrappers import MultitaskWrapper
         >>> from torchmetrics.regression import MeanSquaredError
         >>> from torchmetrics.classification import BinaryAccuracy
         >>>
         >>> classification_target = torch.tensor([0, 1, 0])
         >>> regression_target = torch.tensor([2.5, 5.0, 4.0])
         >>> targets = {"Classification": classification_target, "Regression": regression_target}
         >>>
         >>> classification_preds = torch.tensor([0, 0, 1])
         >>> regression_preds = torch.tensor([3.0, 5.0, 2.5])
         >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
         >>>
         >>> metrics = MultitaskWrapper({
         ...     "Classification": BinaryAccuracy(),
         ...     "Regression": MeanSquaredError()
         ... })
         >>> metrics.update(preds, targets)
         >>> metrics.compute()
         {'Classification': tensor(0.3333), 'Regression': tensor(0.8333)}

    Example (with several metrics per task):
         >>> import torch
         >>> from torchmetrics import MetricCollection
         >>> from torchmetrics.wrappers import MultitaskWrapper
         >>> from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
         >>> from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
         >>>
         >>> classification_target = torch.tensor([0, 1, 0])
         >>> regression_target = torch.tensor([2.5, 5.0, 4.0])
         >>> targets = {"Classification": classification_target, "Regression": regression_target}
         >>>
         >>> classification_preds = torch.tensor([0, 0, 1])
         >>> regression_preds = torch.tensor([3.0, 5.0, 2.5])
         >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
         >>>
         >>> metrics = MultitaskWrapper({
         ...     "Classification": MetricCollection(BinaryAccuracy(), BinaryF1Score()),
         ...     "Regression": MetricCollection(MeanSquaredError(), MeanAbsoluteError())
         ... })
         >>> metrics.update(preds, targets)
         >>> metrics.compute()
         {'Classification': {'BinaryAccuracy': tensor(0.3333), 'BinaryF1Score': tensor(0.)},
          'Regression': {'MeanSquaredError': tensor(0.8333), 'MeanAbsoluteError': tensor(0.6667)}}

    Example (with a prefix and postfix):
        >>> import torch
        >>> from torchmetrics.wrappers import MultitaskWrapper
        >>> from torchmetrics.regression import MeanSquaredError
        >>> from torchmetrics.classification import BinaryAccuracy
        >>>
        >>> classification_target = torch.tensor([0, 1, 0])
        >>> regression_target = torch.tensor([2.5, 5.0, 4.0])
        >>> targets = {"Classification": classification_target, "Regression": regression_target}
        >>> classification_preds = torch.tensor([0, 0, 1])
        >>> regression_preds = torch.tensor([3.0, 5.0, 2.5])
        >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
        >>>
        >>> metrics = MultitaskWrapper({
        ...     "Classification": BinaryAccuracy(),
        ...     "Regression": MeanSquaredError()
        ... }, prefix="train_")
        >>> metrics.update(preds, targets)
        >>> metrics.compute()
        {'train_Classification': tensor(0.3333), 'train_Regression': tensor(0.8333)}

    """

    is_differentiable: bool = False

    def __init__(
        self,
        task_metrics: Dict[str, Union[Metric, MetricCollection]],
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
    ) -> None:
        super().__init__()

        if not isinstance(task_metrics, dict):
            raise TypeError(f"Expected argument `task_metrics` to be a dict. Found task_metrics = {task_metrics}")

        for metric in task_metrics.values():
            if not (isinstance(metric, (Metric, MetricCollection))):
                raise TypeError(
                    "Expected each task's metric to be a Metric or a MetricCollection. "
                    f"Found a metric of type {type(metric)}"
                )

        self.task_metrics = nn.ModuleDict(task_metrics)

        if prefix is not None and not isinstance(prefix, str):
            raise ValueError(f"Expected argument `prefix` to either be `None` or a string but got {prefix}")
        self._prefix = prefix or ""

        if postfix is not None and not isinstance(postfix, str):
            raise ValueError(f"Expected argument `postfix` to either be `None` or a string but got {postfix}")
        self._postfix = postfix or ""


    def items(self, flatten: bool = True) -> Iterable[Tuple[str, nn.Module]]:
        """Iterate over task and task metrics.

        Args:
            flatten: If True, will iterate over all sub-metrics in the case of a MetricCollection.
                If False, will iterate over the task names and the corresponding metrics.

        """
        for task_name, metric in self.task_metrics.items():
            if flatten and isinstance(metric, MetricCollection):
                for sub_metric_name, sub_metric in metric.items():
                    yield f"{self._prefix}{task_name}_{sub_metric_name}{self._postfix}", sub_metric
            else:
                yield f"{self._prefix}{task_name}{self._postfix}", metric

    def keys(self, flatten: bool = True) -> Iterable[str]:
        """Iterate over task names.

        Args:
            flatten: If True, will iterate over all sub-metrics in the case of a MetricCollection.
                If False, will iterate over the task names and the corresponding metrics.

        """
        for task_name, metric in self.task_metrics.items():
            if flatten and isinstance(metric, MetricCollection):
                for sub_metric_name in metric:
                    yield f"{self._prefix}{task_name}_{sub_metric_name}{self._postfix}"
            else:
                yield f"{self._prefix}{task_name}{self._postfix}"

    def values(self, flatten: bool = True) -> Iterable[nn.Module]:
        """Iterate over task metrics.

        Args:
            flatten: If True, will iterate over all sub-metrics in the case of a MetricCollection.
                If False, will iterate over the task names and the corresponding metrics.

        """
        for metric in self.task_metrics.values():
            if flatten and isinstance(metric, MetricCollection):
                yield from metric.values()
            else:
                yield metric

    def update(self, task_preds: Dict[str, Tensor], task_targets: Union[Tensor, Dict[str, Tensor]]) -> None:
        """Update each task's metric with its corresponding pred and target.

        Args:
            task_preds: Dictionary associating each task to a Tensor of pred.
            task_targets: Either a Tensor of target associated with all tasks or a Dictionary associating each task to a Tensor of target.

        ## MODIFY task_targets args so that it also accepts a single target Tensor besides dict of target Tensors
        """

        if isinstance(task_targets, dict) and (not self.task_metrics.keys() == task_preds.keys() == task_targets.keys()):
            raise ValueError(
                "Expected arguments `task_preds` and `task_targets` to have the same keys as the wrapped `task_metrics`"
                f". Found task_preds.keys() = {task_preds.keys()}, task_targets.keys() = {task_targets.keys()} "
                f"and self.task_metrics.keys() = {self.task_metrics.keys()}"
            )
        elif (not isinstance(task_targets, dict)) and (not self.task_metrics.keys() == task_preds.keys()):
            raise ValueError(
                "Expected arguments `task_preds` to have the same keys as the wrapped `task_metrics`"
                f". Found task_preds.keys() = {task_preds.keys()} "
                f"and self.task_metrics.keys() = {self.task_metrics.keys()}"
            )

        for task_name, metric in self.task_metrics.items():
            pred = task_preds[task_name]
            target = task_targets[task_name] if isinstance(task_targets, dict) else task_targets
            metric.update(pred, target)

    def _convert_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the output of the underlying metrics to a dictionary with the task names as keys."""
        return {f"{self._prefix}{task_name}{self._postfix}": task_output for task_name, task_output in output.items()}

    def compute(self) -> Dict[str, Any]:
        """Compute metrics for all tasks."""
        return self._convert_output({task_name: metric.compute() for task_name, metric in self.task_metrics.items()})

    def forward(self, task_preds: Dict[str, Tensor], task_targets: Union[Tensor, Dict[str, Tensor]]) -> Dict[str, Any]:
        """Call underlying forward methods for all tasks and return the result as a dictionary.

        ## MODIFY task_targets args so that it also accepts a single target Tensor besides dict of target Tensors
        """
        # This method is overridden because we do not need the complex version defined in Metric, that relies on the
        # value of full_state_update, and that also accumulates the results. Here, all computations are handled by the
        # underlying metrics, which all have their own value of full_state_update, and which all accumulate the results
        # by themselves.
        return self._convert_output({
            task_name: metric(
                task_preds[task_name],
                task_targets[task_name] if isinstance(task_targets, dict) else task_targets
            )
            for task_name, metric in self.task_metrics.items()
        })

    def reset(self) -> None:
        """Reset all underlying metrics."""
        for metric in self.task_metrics.values():
            metric.reset()
        super().reset()

    @staticmethod
    def _check_arg(arg: Optional[str], name: str) -> Optional[str]:
        if arg is None or isinstance(arg, str):
            return arg
        raise ValueError(f"Expected input `{name}` to be a string, but got {type(arg)}")

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> "MultitaskWrapper":
        """Make a copy of the metric.

        Args:
            prefix: a string to append in front of the metric keys
            postfix: a string to append after the keys of the output dict.

        """
        multitask_copy = deepcopy(self)
        multitask_copy._prefix = self._check_arg(prefix, "prefix") or ""
        multitask_copy._postfix = self._check_arg(postfix, "prefix") or ""
        return multitask_copy

    def plot(
        self, val: Optional[Union[Dict, Sequence[Dict]]] = None, axes: Optional[Sequence[_AX_TYPE]] = None
    ) -> Sequence[_PLOT_OUT_TYPE]:
        """Plot a single or multiple values from the metric.

        All tasks' results are plotted on individual axes.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            axes: Sequence of matplotlib axis objects. If provided, will add the plots to the provided axis objects.
                If not provided, will create them.

        Returns:
            Sequence of tuples with Figure and Axes object for each task.

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.wrappers import MultitaskWrapper
            >>> from torchmetrics.regression import MeanSquaredError
            >>> from torchmetrics.classification import BinaryAccuracy
            >>>
            >>> classification_target = torch.tensor([0, 1, 0])
            >>> regression_target = torch.tensor([2.5, 5.0, 4.0])
            >>> targets = {"Classification": classification_target, "Regression": regression_target}
            >>>
            >>> classification_preds = torch.tensor([0, 0, 1])
            >>> regression_preds = torch.tensor([3.0, 5.0, 2.5])
            >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
            >>>
            >>> metrics = MultitaskWrapper({
            ...     "Classification": BinaryAccuracy(),
            ...     "Regression": MeanSquaredError()
            ... })
            >>> metrics.update(preds, targets)
            >>> value = metrics.compute()
            >>> fig_, ax_ = metrics.plot(value)

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.wrappers import MultitaskWrapper
            >>> from torchmetrics.regression import MeanSquaredError
            >>> from torchmetrics.classification import BinaryAccuracy
            >>>
            >>> classification_target = torch.tensor([0, 1, 0])
            >>> regression_target = torch.tensor([2.5, 5.0, 4.0])
            >>> targets = {"Classification": classification_target, "Regression": regression_target}
            >>>
            >>> classification_preds = torch.tensor([0, 0, 1])
            >>> regression_preds = torch.tensor([3.0, 5.0, 2.5])
            >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
            >>>
            >>> metrics = MultitaskWrapper({
            ...     "Classification": BinaryAccuracy(),
            ...     "Regression": MeanSquaredError()
            ... })
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metrics(preds, targets))
            >>> fig_, ax_ = metrics.plot(values)

        """
        if axes is not None:
            if not isinstance(axes, Sequence):
                raise TypeError(f"Expected argument `axes` to be a Sequence. Found type(axes) = {type(axes)}")

            if not all(isinstance(ax, _AX_TYPE) for ax in axes):
                raise TypeError("Expected each ax in argument `axes` to be a matplotlib axis object")

            if len(axes) != len(self.task_metrics):
                raise ValueError(
                    "Expected argument `axes` to be a Sequence of the same length as the number of tasks."
                    f"Found len(axes) = {len(axes)} and {len(self.task_metrics)} tasks"
                )

        val = val if val is not None else self.compute()
        fig_axs = []
        for i, (task_name, task_metric) in enumerate(self.task_metrics.items()):
            ax = axes[i] if axes is not None else None
            if isinstance(val, Dict):
                f, a = task_metric.plot(val[task_name], ax=ax)
            elif isinstance(val, Sequence):
                f, a = task_metric.plot([v[task_name] for v in val], ax=ax)
            else:
                raise TypeError(
                    "Expected argument `val` to be None or of type Dict or Sequence[Dict]. "
                    f"Found type(val)= {type(val)}"
                )
            fig_axs.append((f, a))
        return fig_axs


class ClasswiseWrapper(WrapperMetric):
    """Wrapper metric for altering the output of classification metrics.

    This metric works together with classification metrics that returns multiple values (one value per class) such that
    label information can be automatically included in the output.

    Args:
        metric: base metric that should be wrapped. It is assumed that the metric outputs a single
            tensor that is split along the first dimension.
        labels: list of strings indicating the different classes.
        prefix: string that is prepended to the metric names.
        postfix: string that is appended to the metric names.

    Example::
        Basic example where the output of a metric is unwrapped into a dictionary with the class index as keys:

        >>> from torch import randint, randn
        >>> from torchmetrics.wrappers import ClasswiseWrapper
        >>> from torchmetrics.classification import MulticlassAccuracy
        >>> metric = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None))
        >>> preds = randn(10, 3).softmax(dim=-1)
        >>> target = randint(3, (10,))
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'multiclassaccuracy_0': tensor(0.5000),
         'multiclassaccuracy_1': tensor(0.7500),
         'multiclassaccuracy_2': tensor(0.)}

    Example::
        Using custom name via prefix and postfix:

        >>> from torch import randint, randn
        >>> from torchmetrics.wrappers import ClasswiseWrapper
        >>> from torchmetrics.classification import MulticlassAccuracy
        >>> metric_pre = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None), prefix="acc-")
        >>> metric_post = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None), postfix="-acc")
        >>> preds = randn(10, 3).softmax(dim=-1)
        >>> target = randint(3, (10,))
        >>> metric_pre(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'acc-0': tensor(0.3333), 'acc-1': tensor(0.6667), 'acc-2': tensor(0.)}
        >>> metric_post(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'0-acc': tensor(0.3333), '1-acc': tensor(0.6667), '2-acc': tensor(0.)}

    Example::
        Providing labels as a list of strings:

        >>> from torch import randint, randn
        >>> from torchmetrics.wrappers import ClasswiseWrapper
        >>> from torchmetrics.classification import MulticlassAccuracy
        >>> metric = ClasswiseWrapper(
        ...    MulticlassAccuracy(num_classes=3, average=None),
        ...    labels=["horse", "fish", "dog"]
        ... )
        >>> preds = randn(10, 3).softmax(dim=-1)
        >>> target = randint(3, (10,))
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'multiclassaccuracy_horse': tensor(0.),
         'multiclassaccuracy_fish': tensor(0.3333),
         'multiclassaccuracy_dog': tensor(0.4000)}

    Example::
        Classwise can also be used in combination with :class:`~torchmetrics.MetricCollection`. In this case, everything
        will be flattened into a single dictionary:

        >>> from torch import randint, randn
        >>> from torchmetrics import MetricCollection
        >>> from torchmetrics.wrappers import ClasswiseWrapper
        >>> from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall
        >>> labels = ["horse", "fish", "dog"]
        >>> metric = MetricCollection(
        ...     {'multiclassaccuracy': ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None), labels),
        ...     'multiclassrecall': ClasswiseWrapper(MulticlassRecall(num_classes=3, average=None), labels)}
        ... )
        >>> preds = randn(10, 3).softmax(dim=-1)
        >>> target = randint(3, (10,))
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'multiclassaccuracy_horse': tensor(0.6667),
         'multiclassaccuracy_fish': tensor(0.3333),
         'multiclassaccuracy_dog': tensor(0.5000),
         'multiclassrecall_horse': tensor(0.6667),
         'multiclassrecall_fish': tensor(0.3333),
         'multiclassrecall_dog': tensor(0.5000)}

    """

    def __init__(
        self,
        metric: Metric,
        labels: Optional[List[str]] = None,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not isinstance(metric, Metric):
            raise ValueError(f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}")
        self.metric = metric

        if labels is not None and not (isinstance(labels, list) and all(isinstance(lab, str) for lab in labels)):
            raise ValueError(f"Expected argument `labels` to either be `None` or a list of strings but got {labels}")
        self.labels = labels

        if prefix is not None and not isinstance(prefix, str):
            raise ValueError(f"Expected argument `prefix` to either be `None` or a string but got {prefix}")
        self._prefix = prefix

        if postfix is not None and not isinstance(postfix, str):
            raise ValueError(f"Expected argument `postfix` to either be `None` or a string but got {postfix}")
        self._postfix = postfix

        self._update_count = 1

    def _filter_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """Filter kwargs for the metric."""
        return self.metric._filter_kwargs(**kwargs)

    def _convert_output(self, x: Tensor) -> Dict[str, Any]:
        """Convert output to dictionary with labels as keys."""
        # Will set the class name as prefix if neither prefix nor postfix is given
        #### MODIFY: change condition (not self.prefix) --> (self._prefix is None) ####
        if self._prefix is None and self._postfix is None:
            prefix = f"{self.metric.__class__.__name__.lower()}_"
            postfix = ""
        else:
            prefix = self._prefix or ""
            postfix = self._postfix or ""
        out = {}
        if self.labels is None:
            out.update({f"{prefix}{i}{postfix}": val for i, val in enumerate(x)})
        else:
            out.update({f"{prefix}{lab}{postfix}": val for lab, val in zip(self.labels, x)})

        #### ADD: add the mean of the metric across all classes ####
        out.update({f"{prefix}{postfix}": x.mean()})
        return out

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate on batch and accumulate to global state."""
        return self._convert_output(self.metric(*args, **kwargs))

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update state."""
        self.metric.update(*args, **kwargs)

    def compute(self) -> Dict[str, Tensor]:
        """Compute metric."""
        return self._convert_output(self.metric.compute())

    def reset(self) -> None:
        """Reset metric."""
        self.metric.reset()

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.wrappers import ClasswiseWrapper
            >>> from torchmetrics.classification import MulticlassAccuracy
            >>> metric = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None))
            >>> metric.update(torch.randint(3, (20,)), torch.randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.wrappers import ClasswiseWrapper
            >>> from torchmetrics.classification import MulticlassAccuracy
            >>> metric = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None))
            >>> values = [ ]
            >>> for _ in range(3):
            ...     values.append(metric(torch.randint(3, (20,)), torch.randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)

    def __getattr__(self, name: str) -> Union[Tensor, nn.Module]:
        """Get attribute from classwise wrapper."""
        if name == "metric" or (name in self.__dict__ and name not in self.metric.__dict__):
            # we need this to prevent from infinite getattribute loop.
            return super().__getattr__(name)

        return getattr(self.metric, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute to classwise wrapper."""
        if hasattr(self, "metric") and name in self.metric._defaults:
            setattr(self.metric, name, value)
        else:
            super().__setattr__(name, value)
            if name == "metric":
                self._defaults = self.metric._defaults
                self._persistent = self.metric._persistent
                self._reductions = self.metric._reductions