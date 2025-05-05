import torch.nn as nn
from typing import Sequence, Literal, Dict
import torch
from torch.nn import functional as F

class MultiResPatchClassifyLoss(nn.Module):
    def __init__(self, mode="classify", patch_res=[16, 32], scale_loss=None, **kwargs):
        super().__init__()
        #assert mode in ["classify", "regress"]
        self.mode = mode
        self.bce = nn.BCEWithLogitsLoss(pos_weight=kwargs.get('binary_pos_wts'))
        self.patch_res = [str(x) for x in patch_res]
        if not scale_loss:
            scale_loss = 1/len(patch_res)
        self.scale_loss = scale_loss

        if self.mode.startswith('multi_label'):
            self.multi_label_bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=kwargs.get('multi_label_pos_wts'))

    def _loss_mse(self, p, y):
        return self.mse(p, y.float())

    def _loss_bce(self, p, y):
        return self.bce(p, y.float())

    def _loss_multi_label_bce(self, p, y):
        return self.multi_label_bce(p, y.float())
    

    def forward(self, patch_preds:dict[str, torch.Tensor | Sequence[torch.Tensor]], patch_trues:dict[str, torch.Tensor | dict]):

        if self.mode in ["binary_classify", "classify"]:
            loss = {"patch_classify_loss": 0.0}

            for patch_res in self.patch_res:
                # for every patch resolution
                pred = patch_preds[patch_res]
                true = patch_trues[patch_res]

                loss[f"patch_classify_bce_loss_res={patch_res}"] = self._loss_bce(pred, true)
                loss["patch_classify_loss"] += loss[
                    f"patch_classify_bce_loss_res={patch_res}"
                ]

            loss["patch_classify_loss"] *= self.scale_loss
                
            return loss

        elif self.mode == "multi_label_and_binary_classify":
            loss = {"binary_classify_loss": 0.0, 'multi_label_classify_loss': 0.0}

            for patch_res in self.patch_res:
                # for every patch resolution

                #Whole Tumor (WT) binary predictions
                loss_term = 'binary_classify_loss'
                WT_pred = patch_preds[patch_res][0]
                WT_true = patch_trues[patch_res]['WT']
                assert WT_pred.shape == WT_true.shape
                loss[f"{loss_term}_res={patch_res}"] = self._loss_bce(pred, true)
                loss[loss_term] += loss[
                    f"{loss_term}_res={patch_res}"
                ]

                #multi_label predictions
                loss_term = 'multi_label_classify_loss'
                multi_label_pred = patch_preds[patch_res][1]
                multi_label_true = patch_trues[patch_res]['tumor_classes']
                assert multi_label_pred.shape == multi_label_true.shape
                loss[f"{loss_term}_loss_res={patch_res}"] = self._loss_multi_label_bce(multi_label_pred, multi_label_true)
                loss[loss_term] += loss[
                    f"{loss_term}_res={patch_res}"
                ]

            loss["binary_classify_loss"] *= self.scale_loss
            loss["multi_label_classify_loss"] *= self.scale_loss

            return loss




class PatchClassifyBCELoss(nn.Module):
    def __init__(self, mode:Literal['binary', 'multilabel']='binary', labels:Sequence[str]=['WT'], pos_weights:Dict[int, Sequence[float]]=None, patch_sizes=[16, 32], scale_loss=None):
        super().__init__()

        num_labels = len(labels)
        if mode == "binary":
            assert num_labels == 1, "num_labels should be 1 for binary classification"

        elif mode == "mulilabel":
            assert num_labels > 1, "num_labels should be greater than 1 for multi-label classification"
        else:
            raise ValueError(f'mode should be either "binary" or "multilabel"')

        self.labels = labels
        self.num_labels = num_labels
        self.mode = mode
        self.loss_term = f"{mode}_patch_classify_bce_loss"

        self.bce = F.binary_cross_entropy_with_logits

        self.pos_weights = None
        if pos_weights is not None:
            assert pos_weights.keys() == patch_sizes
            self.pos_weights = {}
            for patch_size, pos_weight in pos_weights.items():
                assert len(pos_weight) == num_labels, f"pos_weight should be of length {num_labels}"
                self.pos_weights[str(patch_size)] = torch.tensor(pos_weight, dtype=torch.float32).view(num_labels, 1, 1, 1)

        self.patch_sizes = [str(x) for x in patch_sizes]

        if not scale_loss:
            scale_loss = 1/len(patch_sizes)
        self.scale_loss = scale_loss


    def _loss_bce(self, p, y, pos_weight=None):
        # mean over batch, height, width, depth
        return self.bce(p, y.float(), pos_weight=pos_weight, reduction='none').mean(dim=(0, 2, 3, 4))


    def forward(self, preds:dict[str, torch.Tensor], targets:dict[str, torch.Tensor]):
        loss_dict = {self.loss_term: 0.0}

        for patch_size in self.patch_sizes:
            # for every patch resolution
            pred = preds[patch_size]
            target = targets[patch_size]
            assert pred.shape[1] == target.shape[1] == len(self.labels), f"pred {pred.shape}, and target {target.shape} should have {len(self.labels)} channels"

            pos_weight = self.pos_weights[patch_size] if self.pos_weights is not None else None
            loss = self._loss_bce(pred, target, pos_weight=pos_weight)

            for label, loss_label in zip(self.labels, loss):
                loss_dict[f"{self.loss_term}_{label}_res={patch_size}"] = loss_label

            loss_dict[f"{self.loss_term}_res={patch_size}"] = loss.mean()
            loss_dict[self.loss_term] += loss_dict[
                f"{self.loss_term}_res={patch_size}"
            ]

        loss_dict[self.loss_term] *= self.scale_loss
        return loss_dict



