#test MultiPatchClassificationMetrics on random tensors
from torchmetrics.classification import ConfusionMatrix, F1Score, AveragePrecision, MatthewsCorrCoef
import torch
from multi_res_metrics import *


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

def test_segment_metrics():
    #prepare inputs
    shape = (100,3,8,8,8)
    y_true = torch.rand(shape)
    y_pred_16 = complement_randomly_chosen_elements(y_true, frac=0.2)
    y_pred_32 = complement_randomly_chosen_elements(y_true, frac=0.3)
    y_pred_mean = torch.mean(torch.stack([y_pred_16, y_pred_32]), dim=0)
    trues = y_true.gt(0.7)
    preds = {16:y_pred_16, 32:y_pred_32, 'mean':y_pred_mean}

    import time
    start_time = time.time()
    dice = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    dice_mean = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    dice(y_pred_16 > 0.5, trues)
    dice_mean(y_pred_16 > 0.5, trues)
    print('Dice directly on 16 = ' , dice.aggregate())
    print('Dice mean directly on 16 = ' , dice_mean.aggregate())
    dice = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    dice_mean = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    dice(y_pred_32 > 0.5, trues)
    dice_mean(y_pred_32 > 0.5, trues)
    print('Dice directly on 32 = ' , dice.aggregate())
    print('Dice mean directly on 32 = ' , dice_mean.aggregate())
    dice = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    dice_mean = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    dice(y_pred_mean > 0.5, trues)
    dice_mean(y_pred_mean > 0.5, trues)
    print('Dice directly on mean = ' , dice.aggregate())
    print('Dice mean directly on mean = ' , dice_mean.aggregate())
    print('Directly took ', time.time()-start_time)

    start_time = time.time()
    print('Computing Segment metrics on the entire dataset at once.....')
    seg_metric = MultiResSegmentMetrics(sigmoid=False)
    seg_metric(preds, trues)
    all_metrics = seg_metric.compute_metrics()
    print('seg_metrics ', all_metrics)
    print('Once took ', time.time()-start_time)
    
    start_time = time.time()
    print('Computing Segment metrics on the entire dataset iteratively, acculmulating batch-by-batch.....')
    seg_metric = MultiResSegmentMetrics(sigmoid=False)
    batch_size = 10
    for i in range(100//batch_size):
        preds_ = {}
        trues_ = trues[i*batch_size:((i+1)*batch_size)]
        for key in [16, 32, 'mean']:
            pred = preds[key][i*batch_size:((i+1)*batch_size)]
            preds_[key] = pred
        seg_metric(preds_, trues_)
    all_metrics = seg_metric.compute_metrics()
    print('seg_metrics ', all_metrics)
    print('Batch-by-batch took ', time.time()-start_time)


def test_classification_metrics():
    #prepare inputs
    y_true_16 = torch.rand((100,1,8,8,8))
    y_pred_16 = complement_randomly_chosen_elements(y_true_16, frac=0.3)
    y_true_32 = torch.rand((100,1,4,4,4))
    y_pred_32 = complement_randomly_chosen_elements(y_true_32, frac=0.2)
    trues = {16: y_true_16, 32: y_true_32.gt(0.7)}
    preds = {16: y_pred_16, 32: y_pred_32.gt(0.7)}

    import time
    start_time = time.time()
    #Get metrics directly at first
    print('Computing metrics directly at first.....')
    cm = ConfusionMatrix(task='binary',num_classes=2)
    print('CM directly on 16 = ' , cm(y_pred_16 > 0.5, y_true_16))
    cm = ConfusionMatrix(task='binary',num_classes=2)
    print('CM directly on 32 = ' , cm(y_pred_32 > 0.5, y_true_32))

    f1 = F1Score(task='binary',num_classes=2)
    print('F1 directly on 16 = ' , f1(y_pred_16 > 0.5, y_true_16))
    f1 = F1Score(task='binary',num_classes=2)
    print('F1 directly on 32 = ' , f1(y_pred_32 > 0.5, y_true_32))

    ap = AveragePrecision(task='binary',num_classes=2)
    print('AP directly on 16 = ' , ap(y_pred_16, y_true_16))
    ap = AveragePrecision(task='binary',num_classes=2)
    print('AP directly on 32 = ' , ap(y_pred_32, y_true_32))

    mcc = MatthewsCorrCoef(task='binary',num_classes=2)
    print('MCC directly on 16 = ' , mcc(y_pred_16 > 0.5, y_true_16))
    mcc = MatthewsCorrCoef(task='binary',num_classes=2)
    print('MCC directly on 32 = ' , mcc(y_pred_32 > 0.5, y_true_32))
    print('Directly took ', time.time()-start_time)


    start_time = time.time()
    print('Computing Classify metrics on the entire dataset at once.....')
    classify_metrics_names = ['f1', 'ap', 'balanced_accuracy', 'matthews_correlation_coefficient']
    classify_metric = MultiResPatchClassifyMetrics(metrics_names=classify_metrics_names, sigmoid=False)
    classify_metric(preds, trues)
    all_metrics, confmats = classify_metric.compute_metrics()
    print('classify_metrics ', all_metrics)
    print('confmats ', confmats)
    print('Once took ', time.time()-start_time)


    start_time = time.time()
    print('Computing Classify metrics on the entire dataset iteratively, acculmulating batch-by-batch.....')
    classify_metric = MultiResPatchClassifyMetrics(metrics_names=classify_metrics_names, sigmoid=False)
    batch_size = 10
    for i in range(100//batch_size):
        preds_ = {}
        trues_ = {}
        for key in [16, 32]:
            pred = preds[key][i*batch_size:((i+1)*batch_size)]
            true = trues[key][i*batch_size:((i+1)*batch_size)]
            preds_[key] = pred
            trues_[key] = true
        classify_metric(preds_, trues_)
    all_metrics, confmats = classify_metric.compute_metrics()
    print('classify_metrics ', all_metrics)
    print('confmats ', confmats)
    print('Batch-by-batch took ', time.time()-start_time)

if __name__ == "__main__":
        test_segment_metrics()
