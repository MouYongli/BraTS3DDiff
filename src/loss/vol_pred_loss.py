import torch.nn as nn
import torch

class VolumePredLoss(nn.Module):
    def __init__(self):
        super(VolumePredLoss, self).__init__()
        self.mse = nn.MSELoss()

    def _loss_mse(self, p, y):
        return self.mse(p, y.float())


    def forward(self, pred_vols, true_vols):
        assert len(pred_vols) == len(true_vols)
        loss = 0
        for i in range(len(pred_vols)):
            loss+=self._loss_mse(pred_vols[i],true_vols[i])
        return {'loss': loss / len(pred_vols)}

