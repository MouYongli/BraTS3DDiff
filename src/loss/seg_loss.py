import torch
import torch.nn as nn
from torch.nn import functional as F
from monai.losses import MaskedDiceLoss, DiceLoss

class BraTSegLoss(nn.Module):
    def __init__(self, scale_loss:float=1.0, dice_batch=False):
        super().__init__()
        self.dice = DiceLoss(sigmoid=False, batch=dice_batch)
        self.masked_dice = MaskedDiceLoss(sigmoid=False, batch=dice_batch)
        self.bce = F.binary_cross_entropy
        self.scale_loss = scale_loss

    def _loss_dice(self, pred, true, mask=None):
        if mask is not None:
            return self.masked_dice(input=pred, target=true, mask=mask)
        else:
            return self.dice(input=pred, target=true)

    def _loss_bce(self, pred, true, mask=None):
        if mask is not None:
            #calculate loss only on the masked region
            return self.bce(input=pred, target=true.float(), weight=mask) * (mask.numel() / mask.sum())
        else:
            return self.bce(input=pred, target=true.float())


    def forward(self, pred, true, fg=None, prefix=None, suffix=None):
        # p:predicted seg map (probability values)
        # t: true seg map (binary values)
        #fg: brain foreground mask
        assert pred.shape == true.shape 

        dice_loss = self._loss_dice(pred, true, fg)
        bce_loss = self._loss_bce(pred, true, fg)
        loss = self.scale_loss*(dice_loss + bce_loss)

        loss_dict = {
            "dice_loss": dice_loss,
            "bce_loss": bce_loss,
            'loss': loss
        }

        if prefix is not None:
            loss_dict = {f"{prefix}_{k}": v for k, v in loss_dict.items()}

        if suffix is not None:
            loss_dict = {f"{k}_{suffix}": v for k, v in loss_dict.items()}

        return loss_dict, loss


class MultiResSegmentLoss(nn.Module):
    def __init__(self, patch_res:list=[16, 32], incl_mean:bool=True, scale_loss:float=1.0, dice_batch=False):
        super().__init__()
        self.loss_fn = BraTSegLoss(scale_loss=1.0,dice_batch=dice_batch)
        if incl_mean:
            patch_res = patch_res + ['mean']
        self.patch_res = patch_res
        self.scale_loss = scale_loss

    def forward(self,
            preds:dict[torch.Tensor], 
            true:torch.Tensor, 
            fg:torch.Tensor=None, 
            prefix:str='seg'
    ):
        loss_dict = {f'{prefix}_loss' : 0.0}

        for patch_res_ in self.patch_res:
            pred = preds[patch_res_]
            res_loss_dict, res_loss = self.loss_fn(pred, true, fg=fg, prefix=prefix, suffix=f'res={patch_res_}')
            loss_dict[f'{prefix}_loss'] += res_loss
            loss_dict.update(res_loss_dict)

        loss_dict[f'{prefix}_loss'] *= self.scale_loss
        return loss_dict

