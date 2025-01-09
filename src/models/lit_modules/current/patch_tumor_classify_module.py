from typing import Any, List
import os
import json
import torch
from lightning.pytorch import LightningModule

from src.loss.patch_classify_loss import MultiResPatchClassifyLoss
from src.metrics.multi_res_metrics import MultiResPatchClassifyMetrics
from src.models.networks.swinunetr.swinunetr_enc_1x1_conv import SwinUNETREnc128

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PatchTumorClassifyLitModule(LightningModule):
    """LightningModule for Brain Patch Tumor Classification."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        net: SwinUNETREnc128,
        extra_kwargs: dict,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.patch_classify_criterion = MultiResPatchClassifyLoss(
            mode="classify", patch_res=self.hparams.extra_kwargs.patch_sizes, scale_loss=1.0
        )
        self.patch_classify_metric = MultiResPatchClassifyMetrics(patch_sizes=self.hparams.extra_kwargs.patch_sizes,
                                                            sigmoid=True, thresh=self.hparams.extra_kwargs.patch_thresh)


    def forward(self, image: torch.Tensor):
        out = self.net(image)
        patch_embeddings, patch_preds = out["embeddings"], out["labels"]
        return patch_embeddings, patch_preds

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        log.info("Started Training...")

    def training_step(self, batch: Any):
        # image: NxCxWxHxD (C=Channels)
        # mask: NxCxWxHxD  (C=Tumor subregions)
        image, mask, target_patch_labels = (
            batch["image"],
            batch["mask"],
            batch["patch_tumor_labels"],
        )
        _ , patch_logits = self.forward(image)
        loss = self.patch_classify_criterion(patch_logits, target_patch_labels)
        loss['loss'] = loss["patch_classify_loss"]
        
        self.log_scores(
            loss, on_step=True, on_epoch=True, prog_bar=True, prefix="train"
        )
        self.log(
            "global_step", self.global_step, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss["loss"]

    def val_test_step(self, batch, mode="val"):
        # image: BxNx4xWxHxD (4 Image Channels,B=batch size,N=num of sliding windows in an image)
        # mask: BxNxCxWxHxD  (C= Tumor subregions Channels)
        image, mask, target_patch_labels = (
            batch["image"],
            batch["mask"],
            batch["patch_tumor_labels"],
        )
        _ , patch_logits = self.forward(image)
        loss = self.patch_classify_criterion(patch_logits, target_patch_labels)
        loss['loss'] = loss["patch_classify_loss"]
        self.log_scores(loss, prefix=mode, on_epoch=True, prog_bar=True)
        #compute metrics
        self.patch_classify_metric(preds=patch_logits, trues=target_patch_labels)

    def validation_step(self, batch):
        self.val_test_step(batch, mode="val")

    def on_validation_epoch_end(self):
        patch_classify_metrics, confmats = self.patch_classify_metric.compute_metrics()
        patch_classify_metrics['patch_classify_metric'] = 0.5 * (patch_classify_metrics['patch_classify_f1_mean']+\
                                                patch_classify_metrics['patch_classify_ap_mean']
                                            )

        self.log_scores(patch_classify_metrics, prefix="val", on_epoch=True, prog_bar=True)
        log.info(f"Confmats: {json.dumps(confmats)}")


    def test_step(self, batch: Any):
        self.val_test_step(batch, mode="test")

    def on_test_epoch_end(self):
        patch_classify_metrics, confmats = self.patch_classify_metric.compute_metrics()
        patch_classify_metrics['patch_classify_metric'] = 0.5 * (patch_classify_metrics['patch_classify_f1_mean']+\
                                                patch_classify_metrics['patch_classify_ap_mean']
                                            )

        self.log_scores(patch_classify_metrics, prefix="test", on_epoch=True, prog_bar=True)
        log.info(f"Confmats: {json.dumps(confmats)}")

    def log_scores(
        self,
        scores: dict,
        prefix="train",
        on_epoch=None,
        on_step=None,
        prog_bar=False,
    ):
        scores = {f"{prefix}/{k}": v for k, v in scores.items()}
        self.log_dict(scores, on_epoch=on_epoch, on_step=on_step, prog_bar=prog_bar)


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
