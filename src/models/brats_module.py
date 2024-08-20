from typing import Any, List

import torch
from lightning.pytorch import LightningModule
from torchmetrics import MinMetric, MeanMetric
from monai.metrics import DiceMetric
import os
from einops import repeat, rearrange, reduce

from src.loss.brats_loss import BraTSLoss
from monai.inferers import SlidingWindowInferer
from src.models.utils.utils import compute_subregions_pred_metrics

class BraTSLitModule(LightningModule):
    """LightningModule for BraTS segmentation.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        inferer: SlidingWindowInferer
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.inferer = inferer
        self.criterion = BraTSLoss()
        # metric objects for calculating and averaging loss across batches
        # for averaging loss across batches
        self.mean_train_loss = MeanMetric()
        self.mean_val_loss = MeanMetric()
        self.mean_test_loss = MeanMetric()
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True, ignore_empty=False)


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
        image, mask, foreground = batch["image"], batch["mask"], batch["foreground"]
        logits = self.forward(image) * foreground 
        loss = self.criterion(logits, mask)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        loss['loss'] = loss["dice_loss"] + loss["bce_loss"]
        self._log_scores(loss,on_step=True,on_epoch=True,prog_bar=True,prefix='train')
        return {"loss": loss['loss']}


    def val_test_step(self,batch,mode='val'):
        #image: Nx4xWxHxD (4 Image Channels)
        #mask: NxCxWxHxD  (C= Tumor subregions Channels)
        image, mask, foreground = batch["image"], batch["mask"], batch["foreground"]
        N,C,W,H,D = mask.shape
        subregions_names = self.trainer.datamodule.subregions_names
        assert len(subregions_names) == C

        #Evaluate on the entire image using sliding window inference
        logits = self.inferer(inputs=image,network=self.forward) * foreground

        #compute loss on raw logits
        seg_losses = self.criterion(logits, mask)
        seg_losses['loss'] = seg_losses["dice_loss"] + seg_losses["bce_loss"]
        self._log_scores(seg_losses,prefix=mode,on_epoch=True,prog_bar=True)

        #compute scores on binary logits for every subregion
        logits = logits.sigmoid().gt(0.5)
        pred_scores = compute_subregions_pred_metrics(logits,mask,C,subregions_names)
        self._log_scores(pred_scores,prefix=mode,on_epoch=True,prog_bar=True)

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
