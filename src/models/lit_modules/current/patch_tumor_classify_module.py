from typing import Any, List

import torch
from lightning.pytorch import LightningModule
from torchmetrics import MinMetric, MeanMetric
from monai.metrics import DiceMetric
import os
from einops import repeat, rearrange, reduce

from src.loss.brats_loss import BraTSLoss

from src.loss.patch_tumor_loss import PatchTumorLoss
from monai.inferers import SlidingWindowInferer
from src.utils.model_utils import compute_subregions_pred_metrics
import torchmetrics


class PatchTumorClassifyLitModule(LightningModule):
    """LightningModule for Brain Tumor Vol Prediction."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        inferer: SlidingWindowInferer,
        extra_kwargs: dict,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = PatchTumorLoss(
            mode="classify", patch_res=self.hparams.extra_kwargs.patch_sizes
        )
        self.inferer = inferer

        # metric objects for calculating and averaging loss across batches
        # for averaging loss across batches
        self.mean_train_loss = MeanMetric()
        self.mean_val_loss = MeanMetric()
        self.mean_test_loss = MeanMetric()
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor, pred_mode="classify"):
        return self.net(x, pred_mode=pred_mode)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.mean_val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Any):
        # image: NxCxWxHxD (C=Channels)
        # mask: NxCxWxHxD  (C=Tumor subregions)
        image, mask, patch_tumor_labels = (
            batch["image"],
            batch["mask"],
            batch["patch_tumor_labels"],
        )
        pred_patch_tumor_labels = self.forward(image, pred_mode="classify")["labels"]
        loss = self.criterion(pred_patch_tumor_labels, patch_tumor_labels)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.log(
            f"train/loss", loss["loss"], on_step=True, on_epoch=True, prog_bar=True
        )
        self._log_scores(
            loss, on_step=True, on_epoch=True, prog_bar=True, prefix="train"
        )
        return {"loss": loss["loss"]}

    def val_test_step(self, batch, mode="val"):
        # image: BxNx4xWxHxD (4 Image Channels,B=batch size,N=num of sliding windows in an image)
        # mask: BxNxCxWxHxD  (C= Tumor subregions Channels)
        image, mask, patch_tumor_labels = (
            batch["image"],
            batch["mask"],
            batch["patch_tumor_labels"],
        )
        subregions_names = self.trainer.datamodule.subregions_names

        # Evaluate on the entire image using sliding window inference
        # logits = self.inferer(inputs=image,network=self.forward,pred_mode='classify')
        logits = self.forward(image, pred_mode="classify")["labels"]

        # compute loss on raw logits
        loss = self.criterion(logits, patch_tumor_labels)
        self._log_scores(loss, prefix=mode, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            f"{mode}/loss", loss["loss"], on_step=False, on_epoch=True, prog_bar=True
        )

        # compute pred metrics on binarized logits (dice score)
        patch_sizes = self.trainer.datamodule.hparams.patch_sizes
        mean_dice = 0.0
        for patch_size in patch_sizes:
            pred_scores, dice = compute_subregions_pred_metrics(
                logits[patch_size],
                patch_tumor_labels[patch_size],
                1,
                subregions_names,
                suffix_key={"patch_size": patch_size},
            )

            self._log_scores(pred_scores, prefix=mode, on_epoch=True, prog_bar=True)
            mean_dice += dice
        mean_dice /= len(patch_sizes)
        self.log(f"{mode}/dice", mean_dice, on_step=False, on_epoch=True, prog_bar=True)

    def _log_scores(
        self,
        scores: dict,
        prefix="train",
        on_epoch=False,
        on_step=False,
        prog_bar=False,
    ):
        scores = {f"{prefix}/{k}": v for k, v in scores.items()}
        self.log_dict(scores, on_epoch=on_epoch, on_step=on_step, prog_bar=prog_bar)

    def validation_step(self, batch):
        self.val_test_step(batch, mode="val")

    def test_step(self, batch: Any, batch_idx: int):
        self.val_test_step(batch, mode="test")

    def predict_step(self, batch):
        datas, file_ids = batch
        images, foregrounds = datas["image"], datas["foreground"]
        logits = self.inferer(inputs=images, network=self.forward) * foregrounds
        logits = logits.sigmoid().gt(0.5)
        return logits, file_ids

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
