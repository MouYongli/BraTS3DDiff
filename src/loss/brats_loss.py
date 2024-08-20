import torch.nn as nn
from monai.losses import DiceLoss
import torch

class BraTSLoss(nn.Module):
    def __init__(self):
        super(BraTSLoss, self).__init__()
        self.dice = DiceLoss(sigmoid=True, batch=True)
        self.ce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def _loss_dice(self, p, y):
        return self.dice(p, y)
    
    def _loss_ce(self, p, y):
        return self.ce(p, y.float())

    def _loss_mse(self, p, y):
        return self.mse(torch.sigmoid(p), y.float())

    def forward(self, p, y):
        #p:predicted mask
        #y: true mask
        #numchannels of pred_masks is sometimes larger than true mask 
        #for some baseline models trained like that
        assert p.shape[1] >= y.shape[1]

        dice_loss,ce_loss,mse_loss = 0.0,0.0,0.0
        num_channels = y.shape[1]
        for c in range(num_channels):
            p_region = p[:, c].unsqueeze(1)
            y_region = y[:, c].unsqueeze(1)
            dice_loss += self._loss_dice(p_region, y_region)
            ce_loss += self._loss_ce(p_region, y_region)
            mse_loss += self._loss_mse(p_region, y_region)

        dice_loss /= num_channels
        ce_loss /= num_channels
        mse_loss /= num_channels

        return {
            "dice_loss": dice_loss, 
            "bce_loss": ce_loss,
            "mse_loss": mse_loss
        }



