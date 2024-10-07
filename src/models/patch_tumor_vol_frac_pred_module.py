from typing import Any, List

import torch
from lightning.pytorch import LightningModule
from torchmetrics import MinMetric, MeanMetric
from monai.metrics import DiceMetric
import os
from einops import repeat, rearrange, reduce

from BraTS3DDiff.src.loss.patch_tumor_loss import VolumePredLoss
from monai.inferers import SlidingWindowInferer
from src.models.utils.utils import compute_subregions_pred_metrics
from copy import deepcopy
import torchmetrics

class BrainTumorVolPredLitModule(LightningModule):
    """LightningModule for Brain Tumor Vol Prediction.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = VolumePredLoss()
        # metric objects for calculating and averaging loss across batches
        # for averaging loss across batches
        self.mean_train_loss = MeanMetric()
        self.mean_val_loss = MeanMetric()
        self.mean_test_loss = MeanMetric()
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()



    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.mean_val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Any):
        #image: NxCxWxHxD (C=Channels)
        #mask: NxCxWxHxD  (C=Tumor subregions)
        image, mask, volume_maps = batch["image"], batch["mask"], batch["volume_maps"]
        pred_volume_maps= self.forward(image)
        loss = self.criterion(pred_volume_maps, volume_maps)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.log(f"train/loss", loss['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self._log_scores(loss,on_step=True,on_epoch=True,prog_bar=True,prefix='train')
        return {"loss": loss['loss']}

    def val_test_step(self,batch,mode='val'):
        #image: BxNx4xWxHxD (4 Image Channels,B=batch size,N=num of sliding windows in an image)
        #mask: BxNxCxWxHxD  (C= Tumor subregions Channels)
        image, mask, volume_maps = batch["image"], batch["mask"], batch["volume_maps"]
        N,C,W,H,D = mask.shape
        subregions_names = self.trainer.datamodule.subregions_names
        assert len(subregions_names) == C

        #Evaluate on the entire image using sliding window inference
        pred_volume_maps = self.forward(image)

        #compute loss on raw logits
        loss = self.criterion(pred_volume_maps, volume_maps)
        #self._log_scores(seg_losses,prefix=mode,on_step=False,on_epoch=True,prog_bar=True)
        self.log(f"{mode}/loss", loss['loss'], on_step=False, on_epoch=True, prog_bar=True)

        #binarize volume maps
        patch_sizes = self.trainer.datamodule.hparams.patch_sizes
        bin_threshs = [0.0,1e-5,1e-3,1e-2]
        mean_dice = 0.0
        for i in range(len(patch_sizes)):
            volume_maps[i][volume_maps[i]>0] = 1.0
            for thresh in bin_threshs:
                pred_volume_maps_thresh = deepcopy(pred_volume_maps)
                pred_volume_maps_thresh[i][pred_volume_maps_thresh[i] > thresh] = 1.0
                pred_scores, dice = compute_subregions_pred_metrics(pred_volume_maps_thresh[i],volume_maps[i],C,subregions_names,suffix_key={'thresh':thresh,'patch_size':patch_sizes[i]})
                self._log_scores(pred_scores,prefix=mode,on_epoch=True,prog_bar=True)
                mean_dice += dice
        mean_dice /= len(patch_sizes) * len(bin_threshs)
        self.log(f"{mode}/dice", mean_dice, on_step=False, on_epoch=True, prog_bar=True)


    def _log_scores(self,scores:dict,prefix='train',on_epoch=False,on_step=False,prog_bar=False):
        scores = {f"{prefix}/{k}":v for k,v in scores.items()}
        self.log_dict(scores,on_epoch=on_epoch,on_step=on_step,prog_bar=prog_bar)

    def validation_step(self, batch):
        self.val_test_step(batch,mode='val')

    def test_step(self, batch: Any, batch_idx: int):
        self.val_test_step(batch,mode='test')

    def predict_step(self, batch):
        datas, file_ids = batch
        images, foregrounds = datas["image"], datas["foreground"]
        logits = self.inferer(inputs=images,network=self.forward) * foregrounds
        logits = logits.sigmoid().gt(0.5)
        return logits,file_ids


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            print(self.hparams)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
