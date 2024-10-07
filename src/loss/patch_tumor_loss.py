import torch.nn as nn
import torch
from monai.losses import DiceLoss

class PatchTumorLoss(nn.Module):
    def __init__(self,mode='classify',patch_res=[16,32], avg=False):
        super(PatchTumorLoss, self).__init__()
        assert mode in ['classify', 'regress']
        self.mode = mode
        self.dice = DiceLoss(sigmoid=True, batch=True)
        self.ce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.patch_res = patch_res
        self.avg = avg

    def _loss_mse(self, p, y):
        return self.mse(p, y.float())

    def _loss_dice(self, p, y):
        return self.dice(p, y)
    
    def _loss_ce(self, p, y):
        return self.ce(p, y.float())

    def forward(self, patch_preds, patch_trues):
        assert len(patch_preds) == len(patch_trues) == len(self.patch_res) > 1

        #num_channels==1
        assert list(patch_preds.values())[0].shape[1] == list(patch_trues.values())[0].shape[1] == 1

        if self.mode == 'regress':
            loss = {'loss':0.0}
            loss.update({f"loss_res={k}":0.0 for k in self.patch_res})

            for patch_res in self.patch_res:
                #for every patch resolution
                patch_pred = patch_preds[patch_res]
                patch_true = patch_trues[patch_res]
                mse_loss = self._loss_mse(patch_pred,patch_true)
                loss[f"mse_loss_res={patch_res}"] = mse_loss
                loss['loss'] += loss[f"mse_loss_res={patch_res}"]
            return loss

        elif self.mode == 'classify':
            loss = {'patch_classify_loss':0}

            for patch_res in self.patch_res:
                #for every patch resolution
                patch_pred = patch_preds[patch_res]
                patch_true = patch_trues[patch_res]
                dice_loss = self._loss_dice(patch_pred,patch_true)
                ce_loss = self._loss_ce(patch_true,patch_true)
                loss[f"patch_classify_dice_loss_res={patch_res}"] = dice_loss
                loss[f"patch_classify_ce_loss_res={patch_res}"] = ce_loss

                loss[f"patch_classify_loss_res={patch_res}"] = loss[f"patch_classify_dice_loss_res={patch_res}"] + loss[f"patch_classify_ce_loss_res={patch_res}"]
                loss['patch_classify_loss'] += loss[f"patch_classify_loss_res={patch_res}"]

            if self.avg:
                loss['patch_classify_loss'] /= len(self.patch_res)

            return loss

