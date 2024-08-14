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
        assert p.shape == y.shape
        
        return {
            "dice_loss": self._loss_dice(p, y), 
            "bce_loss": self._loss_ce(p, y),
            "mse_loss": self._loss_mse(p, y)
        }
