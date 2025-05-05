import torch
from torch import nn, Tensor
from torchmetrics.metric import Metric
from torchmetrics.classification import (
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
    Specificity,
    PrecisionRecallCurve,
    AveragePrecision,
)

from torchmetrics import MetricCollection

from overrides import override
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Literal, Final, Tuple
from functools import partial
from copy import deepcopy

from src.metrics.metric_wrappers import MultitaskWrapper, ClasswiseWrapper, DiceMetricWrapper



class MultiResBaseMetrics(ABC, nn.Module):
    '''
        Compute metrics on Multiple Patch Resolutions Base Abstract Class
        Accumulate metrics batch-by-batch, aggregate at the end

        At the end of the epoch, the metrics are aggregated, 
            and the final metrics on the entire dataset, are reported 
    '''
    def __init__(self, metrics_names:List[str], summary_metrics_names:List[str], prefix:str="", patch_sizes:List[int]=[16, 32]):
        super().__init__()
        self.metrics_names = metrics_names
        self.summary_metrics_names = summary_metrics_names
        self.metrics_names_2_simple_names = self._metric_names_simple(metrics_names)
        self.simple_names_2_metrics_names = {v:k for k, v in self.metrics_names_2_simple_names.items()}

        self.patch_sizes = self._patch_sizes(patch_sizes) #sort patch sizes (always deterministic order of logging)
        self.prefix = prefix

        self.compute_groups = self._compute_groups()
        #print('compute_groups', self.compute_groups)
        self.metrics = self._metrics()


    def _metrics(self) -> Metric:
        return MultitaskWrapper({
            f'{patch_size}': MetricCollection(
                {
                    name: self.get_metric(name) for name in self.metrics_names
                },
                compute_groups=self.compute_groups
            )
            for patch_size in self.patch_sizes
        })


    def get_metric(self, metric_name: str) -> Metric:
        #override it in the downstream class if required
        return self.get_base_metric(self.metrics_names_2_simple_names[metric_name])

    @abstractmethod
    def get_base_metric(self, metric_name_simple: str) -> Metric:
        #takes in the simple metric name as arg
        #implement it in the downstream class
        raise NotImplementedError("Method 'get_base_metric()' must be implemented by children classes!!")


    def _compute_groups(self) -> Union[bool, list[list[str]]]:
        '''
        Group metrics into the same compute_group if they share the same metric state
        '''
        _groups = {idx:metric_name for idx, metric_name in enumerate(self.metrics_names)}
        compute_groups = []

        for i in range(len(self.metrics_names)):
            _metric_name = _groups.pop(i, None)
            if _metric_name is None:
                continue

            _metric = self.get_base_metric(self.metrics_names_2_simple_names[_metric_name])
            compute_groups.append([_metric_name])
            for j, _cmp_metric_name in deepcopy(_groups).items():                    
                _cmp_metric = self.get_base_metric(self.metrics_names_2_simple_names[_cmp_metric_name])
                if MetricCollection._equal_metric_states(_metric, _cmp_metric):
                    compute_groups[-1].append(_cmp_metric_name)
                    _groups.pop(j)

        return compute_groups



    def _keys2str(self, 
            preds:Dict[str, Tensor],
            trues:Union[Tensor, Dict[str, Tensor]]
        )->Tuple:
        preds = {f'{k}':v for k, v in preds.items()}
        if isinstance(trues, dict):
            trues = {f'{k}':v for k, v in trues.items()}
        return (preds, trues)


    def forward(self, 
            preds:Dict[str, Tensor],
            trues:Union[Tensor, Dict[str, Tensor]],
        ) -> Dict[str, Any]:

        preds, trues = self._keys2str(preds, trues)
        return self._convert_output(self.metrics(preds, trues))


    def update(self, 
            preds:Dict[str, Tensor],
            trues:Union[Tensor, Dict[str, Tensor]],
        ):
        preds, trues = self._keys2str(preds, trues)
        self.metrics.update(preds, trues)


    def compute(self) -> Dict[str, Any]:
        return self._convert_output(self.metrics.compute())

    def reset(self) -> None:
        self.metrics.reset()
    
    def clone(self, prefix="") -> "MultiResBaseMetrics":
        metrics_copy = deepcopy(self)
        metrics_copy.prefix = f'{prefix}{self.prefix}'
        return metrics_copy


    def _add_patchsize_metrics(self, metric_name:str, out, flat_out:Dict[str, Any]={}, mean:bool=True) -> Dict[str, Any]:
        #Add a certain metric across all patch_sizes to flat_out
        #Average metric values across patch sizes (if mean=True)
        new_metric_name = f"{self.prefix}{metric_name.rstrip('_')}_"
        if mean:
            mean_metric = 0.0
        for patch_size in self.patch_sizes:
            metric_val = out[patch_size][metric_name]
            flat_out.update({f"{new_metric_name}res={patch_size}": metric_val})
            if mean:
                mean_metric += metric_val
        if mean:
            mean_metric /= len(self.patch_sizes)
            flat_out.update({f"{new_metric_name}mean": mean_metric})
        return flat_out


    def _convert_output(self, out: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        '''
            - Postprocess metric names
            - Average metrics  across patch sizes
            - Flatten metrics output dict
        '''

        flat_out = {}
        metrics_names = sorted(out[self.patch_sizes[0]].keys()) #sort metric names (deterministic order of logging)
        for metric_name in metrics_names:
            flat_out = self._add_patchsize_metrics(metric_name, out, flat_out, mean=True)

        flat_out = self._metrics_summary(flat_out)
        return flat_out


    def _metrics_summary(self, flat_out: Dict[str, Any]) -> Dict[str, Any]:
        summary_metric = 0.0
        for metric_name in self.summary_metrics_names:
            summary_metric += flat_out[f"{self.prefix}{metric_name}_mean"]
        summary_metric /= len(self.summary_metrics_names)
        flat_out.update({f"{self.prefix}_metric": summary_metric})
        return flat_out


    def check_metric_name(self, metric_name: str) -> str:
        """
        There are many metrics and some of them have more than one names. 
        In addition, some of the names are very long.
        Therefore, this function is used to check and simplify the name.

        Returns:
            Simplified metric name.

        Raises:
            NotImplementedError: when the metric is not implemented.
        """
        
        metric_name = metric_name.replace(" ", "_")
        metric_name = metric_name.lower()
        #implement the rest in downstream classes
        return metric_name

    def _metric_names_simple(self, metrics_names: List[str]) -> Dict[str, str]:
        return {name: self.check_metric_name(name) for name in metrics_names}

    def _patch_sizes(self, patch_sizes: List[int]) -> List[str]:
        assert all(isinstance(x, int) for x in patch_sizes), "All patch_sizes must be of type int!!"
        return [f'{patch_size}' for patch_size in sorted(patch_sizes)] #sort patch sizes (always deterministic order of logging)



class PatchClassifyBaseMetrics(MultiResBaseMetrics):

    default_metrics_names: Final = ['confusion_matrix' , 'f1', 'balanced_accuracy', 'precision', 'recall', 'specificity', 'average_precision', 'precision_recall_curve']

    def __init__(self, task:Literal['binary', 'multilabel'], metrics_names:List[str] = None, summary_metrics_names:List[str] = None, patch_sizes:List[int] = [16, 32], **kwargs):
        if metrics_names is None:
            metrics_names = self.default_metrics_names
        
        self.task = task
        self._preprocess_kwargs(**kwargs)

        super().__init__(metrics_names=metrics_names,
                         summary_metrics_names=summary_metrics_names,
                         prefix=f'patch_classify_{task}_',
                         patch_sizes=patch_sizes)

        if not self.summary_metrics_names:
            self.summary_metrics_names = [self.simple_names_2_metrics_names[k] for k in self.summary_metrics_simple_names if k in self.simple_names_2_metrics_names.keys()]

        self.exclude_metrics_names = [self.simple_names_2_metrics_names[k] for k in self.exclude_metrics if k in self.simple_names_2_metrics_names.keys()]


    def _preprocess_kwargs(self, **kwargs):
        self.num_labels = kwargs.get('num_labels')
        self.threshold = kwargs.get('threshold', 0.7)
        self.pr_curve_thresholds = kwargs.get('pr_curve_thresholds', 100)

    @property
    def summary_metrics_simple_names(self)->List[str]:
        return ['f1', 'ap']

    @property
    def exclude_metrics(self) -> List[str]:
        return ['cm', 'pr_curve']

    def _is_exclude_metric(self, metric_name:str)->bool:
        for exclude_metric_name in self.exclude_metrics_names:
            if metric_name.startswith(exclude_metric_name):
                return True
        return False

    @property
    def cm_metrics(self):
        return ['cm', 'f1', 'ba', 'ppv', 'tpr', 'tnr']

    @property
    def pr_curve_metrics(self):
        return ['pr_curve', 'ap']


    @override
    def check_metric_name(self, metric_name: str) -> str:
        """
        There are many metrics and some of them have more than one names. 
        In addition, some of the names are very long.
        Therefore, this function is used to check and simplify the name.

        Returns:
            Simplified metric name.

        Raises:
            NotImplementedError: when the metric is not implemented.
        """
        metric_name = super().check_metric_name(metric_name)

        if metric_name in ['confusion_matrix', 'conf_mat', 'confmat', 'cm']:
            return 'cm'
        if metric_name in ["sensitivity", "recall", "true_positive_rate", "tpr"]:
            return "tpr"
        elif metric_name in ["specificity", "true_negative_rate", "tnr"]:
            return "tnr"
        elif metric_name in ["precision", "positive_predictive_value", "ppv"]:
            return "ppv"
        elif metric_name in ["f1", "f1_score"]:
            return "f1"
        elif metric_name in ["balanced_accuracy", 'bal_acc', "ba"]:
            return "ba"
        elif metric_name in ["precision_recall_curve", 'pr_curve']:
            return "pr_curve"
        elif metric_name in ["average_precision", 'ap']:
            return "ap"
        else:
            raise ValueError(f"Metric {metric_name} is not supported")  


    def _create_metric_kwargs(self, metric_name_simple):
        kwargs = {'task': self.task, 'num_labels':self.num_labels}
        if (self.task == 'multilabel') and (metric_name_simple not in self.exclude_metrics):
            kwargs.update({'average': None})

        if metric_name_simple in self.cm_metrics:
            kwargs.update({'threshold': self.threshold})

        if metric_name_simple in self.pr_curve_metrics:
            kwargs.update({'thresholds': self.pr_curve_thresholds})
        return kwargs


    def get_base_metric(self, metric_name_simple: str) -> Metric:
        kwargs = self._create_metric_kwargs(metric_name_simple)

        if metric_name_simple == 'cm':
            return ConfusionMatrix(**kwargs)

        elif metric_name_simple == 'f1':
            return F1Score(**kwargs)

        elif metric_name_simple == 'ppv':
            return Precision(**kwargs)

        elif metric_name_simple == 'tpr':
            return Recall(**kwargs)

        elif metric_name_simple == 'tnr':
            return Specificity(**kwargs)

        elif metric_name_simple == 'ba':
            return 0.5*(self.get_base_metric('tpr') + self.get_base_metric('tnr'))

        elif metric_name_simple == 'pr_curve':
            return PrecisionRecallCurve(**kwargs)

        elif metric_name_simple == 'ap':
            return AveragePrecision(**kwargs)

        else:
            raise ValueError(f"Metric {metric_name_simple} is not supported")

        

    @override
    def _convert_output(self, out: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        '''
        1. Separate output metrics into main_metrics and other_metrics

            - main_metrics are the metrics that are logged using self.log_dict() in the LightningModule
                --so every metric output needs to be a scalar, and these metrics are averaged across patch sizes
 
            - other_metrics are the metrics defined in self.exclude_metrics (eg confmat/ pr_curve)),
                the outputs of these metrics are non-scalars like multi-dim tensors or tuple of tensors
                DOES NOT need to be logged using self.log_dict() in the LightningModulem, and NOT averaged across patch-sizes

        2. Also, Postprocess metric names and Flatten metrics output dict
        '''

        main_metrics = {}
        other_metrics = {}
        metrics_names = sorted(out[self.patch_sizes[0]].keys()) #sort metric names (deterministic order of logging)
        for metric_name in metrics_names:
    
            if not self._is_exclude_metric(metric_name):
                main_metrics = self._add_patchsize_metrics(metric_name, out, main_metrics, mean=True)
            else:
                other_metrics = self._add_patchsize_metrics(metric_name, out, other_metrics, mean=False)

        main_metrics = self._metrics_summary(main_metrics)

        return {'main_metrics': main_metrics, 'other_metrics': other_metrics}



class MultiLabelPatchClassifyMetrics(PatchClassifyBaseMetrics):

    def __init__(self, labels:List[str]=['ED', 'ET', 'NCR'], metrics_names:List[str] = None, summary_metrics_names:List[str] = None, patch_sizes:List[int] = [16, 32], threshold:float=0.7, pr_curve_thresholds=10):
        self.labels = labels

        super().__init__(
            task='multilabel',
            metrics_names=metrics_names,
            summary_metrics_names=summary_metrics_names,
            patch_sizes=patch_sizes,
            num_labels=len(self.labels),
            threshold=threshold,
            pr_curve_thresholds=pr_curve_thresholds,       
    )

    @override
    def get_metric(self, metric_name:str) -> Metric:
        metric_name_simple = self.metrics_names_2_simple_names[metric_name]
        metric = self.get_base_metric(metric_name_simple)

        if metric_name_simple in self.exclude_metrics:
            #don't wrap an exclude_metric like cm or pr_curve
            return metric

        return ClasswiseWrapper(metric, labels=self.labels, prefix=f'{metric_name}_')




class BinaryPatchClassifyMetrics(PatchClassifyBaseMetrics):
    def __init__(self, metrics_names:List[str] = None, summary_metrics_names:List[str] = None, patch_sizes:List[int] = [16, 32], threshold:float=0.7, pr_curve_thresholds=10):

        super().__init__(
            task='binary',
            metrics_names=metrics_names,
            summary_metrics_names=summary_metrics_names,
            patch_sizes=patch_sizes,
            threshold=threshold,
            pr_curve_thresholds=pr_curve_thresholds
    )



class SegmentationMetrics(MultiResBaseMetrics):
    default_metrics_names: Final = ['dice']

    def __init__(self, metrics_names:List[str] = None, summary_metrics_names:List[str] = None, patch_sizes:List[int] = [16, 32], incl_mean:bool=False, labels=['WT','TC','ET'], threshold:float=0.5):
        if metrics_names is None:
            metrics_names = self.default_metrics_names

        self.incl_mean = incl_mean
        self.labels = labels
        self.threshold = threshold

        super().__init__(metrics_names=metrics_names,
                         summary_metrics_names=summary_metrics_names,
                         prefix=f'seg_',
                         patch_sizes=patch_sizes)


    def get_base_metric(self, metric_name_simple: str) -> Metric:
        if metric_name_simple == 'dice':
            return DiceMetricWrapper(sigmoid=self.sigmoid, threshold=self.threshold)
        else:
            raise ValueError(f"Metric {metric_name_simple} is not supported")

    @override
    def get_metric(self, metric_name:str) -> Metric:
        metric = super().get_metric(metric_name)
        return ClasswiseWrapper(metric, labels=self.labels, prefix=f"{metric_name}_")

    @override
    def check_metric_name(self, metric_name: str) -> str:
        """
        There are many metrics and some of them have more than one names. 
        In addition, some of the names are very long.
        Therefore, this function is used to check and simplify the name.

        Returns:
            Simplified metric name.

        Raises:
            ValueError: when the metric is not implemented.
        """
        metric_name = super().check_metric_name(metric_name)

        if metric_name in ['dice_metric', 'dice']:
            return 'dice'
        else:
            raise ValueError(f"Metric {metric_name} is not supported")


    @override
    def _patch_sizes(self, patch_sizes: List[int]) -> List[str]:
        patch_sizes = super()._patch_sizes(patch_sizes)
        if self.incl_mean:
            patch_sizes += ['mean']
        return patch_sizes



if __name__ == "__main__":
    import time
    from pprint import pprint

    def complement_randomly_chosen_elements(y_true, frac=0.3):
        assert ((y_true >= 0) & (y_true <= 1)).all()
        #create y_pred by randomly swapping x% of y_true values
        y_pred = y_true.detach().clone()
        num_elements = y_true.numel()
        num_samples = int(num_elements * frac)  # x% of the elements
        #swap randomly chosen indices
        idxs = torch.randperm(num_elements)[:num_samples]
        idxs = torch.unravel_index(idxs,shape=y_true.shape)
        y_pred[idxs] = 1 - y_pred[idxs]
        return y_pred

    def _print(text, x, n=30):
        print("-"*n)
        print(text)
        print("-"*n)
        pprint(x, sort_dicts=False)
        print('\n')

    #create dummy inputs
    def create_patch_classify_preds_and_target(**kwargs):
        patch_sizes = kwargs.get('patch_sizes', [8, 16, 32])
        fracs = kwargs.get('fracs', {8: 0.1, 16: 0.2, 32: 0.3})
        im_size = kwargs.get('im_size', 128)
        n_channels = kwargs.get('n_channels', 3)
        batch_size = kwargs.get('batch_size', 100)
        thresh = kwargs.get('thresh', 0.6)
        preds = {}
        targets = {}
        patch_sizes = sorted(patch_sizes)
        for patch_size in patch_sizes:
            target = torch.rand((batch_size, n_channels, im_size//patch_size, im_size//patch_size, im_size//patch_size))
            pred = complement_randomly_chosen_elements(target, frac=fracs[patch_size])
            preds[patch_size] = pred
            targets[patch_size] = target.gt(thresh)
        return preds, targets


    preds, targets = create_patch_classify_preds_and_target(n_channels=1, batch_size=500)
    print('preds', {k:v.shape for k,v in preds.items()})
    print('targets', {k:v.shape for k,v in targets.items()})

    patch_classify_metrics = BinaryPatchClassifyMetrics(patch_sizes=[8, 32, 16], threshold=0.5)
    _print('multilabel_patch_classify_metrics', patch_classify_metrics)

    start_time = time.time()
    patch_classify_metrics.update(preds, targets)
    out = patch_classify_metrics.compute()
    _print('main_metrics:', out['main_metrics'])
    _print('other_metrics:', out['other_metrics'])
    print('Forward took ', time.time()-start_time)
    patch_classify_metrics.reset() #always reset if metric needs to be ready for new data


    '''
    preds, targets = create_patch_classify_preds_and_target(n_channels=3, batch_size=500)
    print('preds', {k:v.shape for k,v in preds.items()})
    print('targets', {k:v.shape for k,v in targets.items()})

    patch_classify_metrics = MultiLabelPatchClassifyMetrics(patch_sizes=[8, 32, 16], labels=['ED', 'ET', 'NCR'], threshold=0.5)
    _print('multilabel_patch_classify_metrics', patch_classify_metrics)

    start_time = time.time()
    patch_classify_metrics.update(preds, targets)
    out = patch_classify_metrics.compute()
    _print('main_metrics:', out['main_metrics'])
    _print('other_metrics:', out['other_metrics'])
    print('Forward took ', time.time()-start_time)
    patch_classify_metrics.reset() #always reset if metric needs to be ready for new data
    '''


