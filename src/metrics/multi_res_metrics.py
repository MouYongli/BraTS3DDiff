
from torchmetrics.classification import BinaryConfusionMatrix, BinaryAveragePrecision, BinaryAUROC
import torch
from torch import nn
from monai.metrics import DiceMetric, CumulativeIterationMetric
from torchmetrics.metric import Metric
from overrides import override

class MultiResBaseMetrics(nn.Module):
    '''
        Compute metrics on Multiple Patch Resolutions Base Class
        Accumulate metrics batch-by-batch, aggregate at the end

        At the end of the epoch, the metrics are aggregated, 
            and the final metrics on the entire dataset, are reported 
    '''
    def __init__(self, metrics_names: list, task_name:str, patch_sizes=[16,32], sigmoid=True, thresh=0.5, no_binarize:list=[]):
        super().__init__()

        self.patch_sizes = patch_sizes
        self.sigmoid = sigmoid
        self.thresh = thresh
        self.task_name = task_name
        self.metrics_names = metrics_names
        self.metrics = self._get_metrics_objs_dict(patch_sizes, metrics_names)
        self.no_binarize = no_binarize #some metrics like AP, AUCROC need to be computed on raw prob scores

    @classmethod
    def _get_metrics_objs_dict(cls, patch_sizes, metrics_names):
        return {patch_size: {name: cls._get_metric_obj(name) for name in metrics_names} for patch_size in patch_sizes}

    @staticmethod
    def _get_metric_obj(metric_name):
        #implement it in the downstream class
        return None

    @staticmethod
    def _flatten_dict(x: dict[dict]):
        '''
            x = {
                'patch_size_1': {
                    'metric_1': xxx1,
                    'metric_2': yyy1,
                        ....
                },
                'patch_size_2': {
                    'metric_1': xxx2,
                    'metric_2': yyy2,
                        ....
                },
                    .......
            }
            
        Returns: {
            '{metric_1}_res={patch_size_1} : xxx1,
            '{metric_2}_res={patch_size_1} : yyy1,
            '{metric_1}_res={patch_size_2} : xxx2,
            '{metric_2}_res={patch_size_2} : yyy2,
                ........
        }
        '''
        flat_dict = {}
        for outer_key in x.keys():
            for inner_key, inner_val in x[outer_key].items():
                flat_dict[f"{inner_key}_res={outer_key}"] = inner_val
        return flat_dict

    @staticmethod
    def _rename_keys(x:dict, prefix):
        x_ = {}
        for k, v in x.items():
            x_[f"{prefix}_{k}"] = v
        return x_

    @staticmethod
    def _calc_metrics_mean(metrics_dict):
        #compute the mean of all metrics across patchsizes
        mean_metrics = {}
        patch_sizes = list(metrics_dict.keys())
        metrics_names = list(metrics_dict[patch_sizes[0]].keys())
        for metric_name in metrics_names:
            metric_name_ = f"{metric_name}_mean"
            for patch_size in patch_sizes:
                mean_metrics[metric_name_] = mean_metrics.get(metric_name_, 0.0) + metrics_dict[patch_size][metric_name]
            mean_metrics[metric_name_] /= len(patch_sizes)
        return mean_metrics

    def forward(self, 
            preds:dict[torch.Tensor],
            trues:torch.Tensor | dict[torch.Tensor],
            masks:torch.Tensor | dict[torch.Tensor] = None
        ):

        #per-batch compute and accumulate
        #preds need to be logits or prob values in [0,1]
        for patch_size in self.patch_sizes:
            true = trues[patch_size] if isinstance(trues, dict) else trues
            pred = preds[patch_size]
            assert true.shape == pred.shape
            if self.sigmoid:
                pred = pred.sigmoid()
            if masks is not None:
                mask = masks[patch_size] if isinstance(masks, dict) else masks
                pred = pred * mask
            #compute metrics
            for metric_name, metric_obj in self.metrics[patch_size].items():
                if metric_name in self.no_binarize:
                    metric_obj(pred, true)
                else:
                    metric_obj(pred.gt(self.thresh), true)

    def aggregate_metrics(self):
        #at epoch end, aggrgate batch wise computed confmats and reset it
        metrics_dict = {}
        for patch_size in self.patch_sizes:
            metrics_dict[patch_size] = {}
            for metric_name, metric_obj in self.metrics[patch_size].items(): 
                if isinstance(metric_obj, Metric):
                    metric_val = metric_obj.compute()
                elif isinstance(metric_obj, CumulativeIterationMetric):
                    metric_val = metric_obj.aggregate()
                else:
                    raise ValueError('Metric should be of type torchmetrics.metric.Metric or monai.metrics.CumulativeIterationMetric!')
                if metric_val.numel() == 1:
                    metric_val = metric_val.item()
                metrics_dict[patch_size][metric_name] = metric_val

                #Reset metric for next epoch Important!!
                metric_obj.reset()

        return metrics_dict

    def _postprocess(self, metrics_dict):
        #compute the mean of all metrics across patchsizes
        mean_metrics = self._calc_metrics_mean(metrics_dict)
        metrics_dict = self._flatten_dict(metrics_dict)
        metrics_dict.update(mean_metrics)
        metrics_dict = self._rename_keys(metrics_dict, prefix=self.task_name)
        return metrics_dict

    def compute_metrics(self):
        #should be called at epoch end to gather accumulated metrics and output final metrics
        metrics_dict = self.aggregate_metrics()
        return self._postprocess(metrics_dict)



class MultiResSegmentMetrics(MultiResBaseMetrics):
    def __init__(self, metrics_names: list=None, patch_sizes=[16,32], incl_mean=True, channels=['WT','TC','ET'], sigmoid=True, thresh=0.5):
        self.default_metrics_names = ['dice']
        if metrics_names is None:
            metrics_names=self.default_metrics_names
        if incl_mean:
            patch_sizes = patch_sizes + ['mean']
        super().__init__(metrics_names=metrics_names,
                         task_name='seg',
                         patch_sizes=patch_sizes,
                         sigmoid=sigmoid,
                         thresh=thresh)

        self.channels = channels

    @staticmethod
    def _get_metric_obj(metric):
        if metric == 'dice':
            return DiceMetric(
                include_background=True,
                reduction="mean_batch",
                ignore_empty=False,
            )
        else:
            raise ValueError('Metric not recognized')

    @override
    def compute_metrics(self):
        metrics_dict = self.aggregate_metrics()

        for patch_size in self.patch_sizes:
            for metric_name in self.metrics_names:
                metric_val = metrics_dict[patch_size].pop(metric_name)
                assert metric_val.shape[0] == len(self.channels)
                metrics_mean = metric_val.mean()
                metrics_dict[patch_size][metric_name] = metrics_mean
                for i, channel in enumerate(self.channels):
                    metrics_dict[patch_size][f"{metric_name}_{channel}"] = metric_val[i] #store channel wise metrics

        return self._postprocess(metrics_dict)
    


class MultiResPatchClassifyMetrics(MultiResBaseMetrics):
    '''
        Compute binary (tumor/non-tumor) patch classification metrics on Multiple Patch Resolutions
        Metrics are computed using forward() and accumulated every batch
        C
        Confmat metrics like F1, Recall etc are computed directly from the conf mat at the end

        At the end of the epoch, the metrics are aggregated, 
            and the final metrics on the entire dataset, are reported 
    '''
    def __init__(self, metrics_names: list=None, patch_sizes=[16,32], sigmoid=True, thresh=0.7):
        self.all_confmat_metrics_names = ['f1', 'balanced_accuracy', 'precision', 'recall', 'specificity', 'matthews_correlation_coefficient']
        self.default_metrics_names = self.all_confmat_metrics_names + ['ap', 'aucroc']
        if metrics_names is None:
            metrics_names=self.default_metrics_names
        #metrics like f1, recall that can be computed directly from the confmat
        self.confmat_metrics_names = set(metrics_names).intersection(set(self.all_confmat_metrics_names))

        #main metrics are computed every batch, confmat metrics are computed directly from the confmat at end
        main_metrics = set(metrics_names).difference(self.confmat_metrics_names)
        main_metrics.add('confmat') #{'confmat','auc','ap'}

        super().__init__(metrics_names=main_metrics,
                         task_name='patch_classify',
                         patch_sizes=patch_sizes,
                         sigmoid=sigmoid,
                         thresh=thresh,
                         no_binarize=['ap', 'aucroc'])

    @staticmethod
    def _get_metric_obj(metric):
        if metric == 'cm' or metric == 'confmat':
            return BinaryConfusionMatrix()
        elif metric == 'ap':
            return BinaryAveragePrecision()
        elif metric == 'aucroc':
            return BinaryAUROC()
        else:
            raise ValueError('Metric not recognized')

    @override
    def compute_metrics(self):
        #should be called at epoch end to gather accumulated metrics and output final metrics
        metrics_dict = self.aggregate_metrics()
        confmats = {}

        for patch_size in self.patch_sizes:
            confmat = metrics_dict[patch_size].pop('confmat').flatten()
            confmats[patch_size] = confmat
            for cm_metric in self.confmat_metrics_names:
                metrics_dict[patch_size][cm_metric] = compute_confusion_matrix_metric(cm_metric, confmat).item()

        metrics_dict = self._postprocess(metrics_dict) #compute patchwise metric mean and flatten dict

        #pretty confmats
        for patch_size in self.patch_sizes:
            confmats[patch_size] = dict(zip(['tn','fp','fn','tp'], confmats[patch_size].tolist()))

        return metrics_dict, confmats



def compute_confusion_matrix_metric(metric_name: str, confusion_matrix: torch.Tensor) -> torch.Tensor:
    """
    This function is used to compute confusion matrix related metric.

    Args:
        metric_name: [``"sensitivity"``, ``"specificity"``, ``"precision"``, ``"negative predictive value"``,
            ``"miss rate"``, ``"fall out"``, ``"false discovery rate"``, ``"false omission rate"``,
            ``"prevalence threshold"``, ``"threat score"``, ``"accuracy"``, ``"balanced accuracy"``,
            ``"f1 score"``, ``"matthews correlation coefficient"``, ``"fowlkes mallows index"``,
            ``"informedness"``, ``"markedness"``]
            Some of the metrics have multiple aliases (as shown in the wikipedia page aforementioned),
            and you can also input those names instead.
        confusion_matrix: Please see the doc string of the function ``get_confusion_matrix`` for more details.

    Raises:
        ValueError: when the size of the last dimension of confusion_matrix is not 4.
        NotImplementedError: when specify a not implemented metric_name.

    """

    metric = check_confusion_matrix_metric_name(metric_name)

    input_dim = confusion_matrix.ndimension()
    if input_dim == 1:
        confusion_matrix = confusion_matrix.unsqueeze(dim=0)
    if confusion_matrix.shape[-1] != 4:
        raise ValueError("the size of the last dimension of confusion_matrix should be 4.")
    #tn,fp,fn,tp
    tn = confusion_matrix[..., 0]
    fp = confusion_matrix[..., 1]
    fn = confusion_matrix[..., 2]
    tp = confusion_matrix[..., 3]
    p = tp + fn
    n = fp + tn
    # calculate metric
    numerator: torch.Tensor
    denominator: torch.Tensor | float
    nan_tensor = torch.tensor(float("nan"), device=confusion_matrix.device)
    if metric == "tpr":
        numerator, denominator = tp, p
    elif metric == "tnr":
        numerator, denominator = tn, n
    elif metric == "ppv":
        numerator, denominator = tp, (tp + fp)
    elif metric == "npv":
        numerator, denominator = tn, (tn + fn)
    elif metric == "fnr":
        numerator, denominator = fn, p
    elif metric == "fpr":
        numerator, denominator = fp, n
    elif metric == "fdr":
        numerator, denominator = fp, (fp + tp)
    elif metric == "for":
        numerator, denominator = fn, (fn + tn)
    elif metric == "pt":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator = torch.sqrt(tpr * (1.0 - tnr)) + tnr - 1.0
        denominator = tpr + tnr - 1.0
    elif metric == "ts":
        numerator, denominator = tp, (tp + fn + fp)
    elif metric == "acc":
        numerator, denominator = (tp + tn), (p + n)
    elif metric == "ba":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator, denominator = (tpr + tnr), 2.0
    elif metric == "f1":
        numerator, denominator = tp * 2.0, (tp * 2.0 + fn + fp)
    elif metric == "mcc":
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    elif metric == "fm":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
        numerator = torch.sqrt(ppv * tpr)
        denominator = 1.0
    elif metric == "bm":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator = tpr + tnr - 1.0
        denominator = 1.0
    elif metric == "mk":
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
        npv = torch.where((tn + fn) > 0, tn / (tn + fn), nan_tensor)
        numerator = ppv + npv - 1.0
        denominator = 1.0
    else:
        raise NotImplementedError("the metric is not implemented.")

    if isinstance(denominator, torch.Tensor):
        return torch.where(denominator != 0, numerator / denominator, nan_tensor)
    return numerator / denominator

def check_confusion_matrix_metric_name(metric_name: str) -> str:
    """
    There are many metrics related to confusion matrix, and some of the metrics have
    more than one names. In addition, some of the names are very long.
    Therefore, this function is used to check and simplify the name.

    Returns:
        Simplified metric name.

    Raises:
        NotImplementedError: when the metric is not implemented.
    """
    metric_name = metric_name.replace(" ", "_")
    metric_name = metric_name.lower()
    if metric_name in ["sensitivity", "recall", "hit_rate", "true_positive_rate", "tpr"]:
        return "tpr"
    if metric_name in ["specificity", "selectivity", "true_negative_rate", "tnr"]:
        return "tnr"
    if metric_name in ["precision", "positive_predictive_value", "ppv"]:
        return "ppv"
    if metric_name in ["negative_predictive_value", "npv"]:
        return "npv"
    if metric_name in ["miss_rate", "false_negative_rate", "fnr"]:
        return "fnr"
    if metric_name in ["fall_out", "false_positive_rate", "fpr"]:
        return "fpr"
    if metric_name in ["false_discovery_rate", "fdr"]:
        return "fdr"
    if metric_name in ["false_omission_rate", "for"]:
        return "for"
    if metric_name in ["prevalence_threshold", "pt"]:
        return "pt"
    if metric_name in ["threat_score", "critical_success_index", "ts", "csi"]:
        return "ts"
    if metric_name in ["accuracy", "acc"]:
        return "acc"
    if metric_name in ["balanced_accuracy", "ba"]:
        return "ba"
    if metric_name in ["f1_score", "f1"]:
        return "f1"
    if metric_name in ["matthews_correlation_coefficient", "mcc"]:
        return "mcc"
    if metric_name in ["fowlkes_mallows_index", "fm"]:
        return "fm"
    if metric_name in ["informedness", "bookmaker_informedness", "bm", "youden_index", "youden"]:
        return "bm"
    if metric_name in ["markedness", "deltap", "mk"]:
        return "mk"
    raise NotImplementedError("the metric is not implemented.")
    
    