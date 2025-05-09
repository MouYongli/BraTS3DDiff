from typing import Any, List
import os
import json
import torch
from lightning.pytorch import LightningModule

from src.loss.patch_classify_loss import PatchClassifyBCELoss
from src.metrics.multi_res_metrics_new import MultiLabelPatchClassifyMetrics, BinaryPatchClassifyMetrics
from src.models.networks.swinunetr.swinunetr_enc_1x1_conv import SwinUNETREnc128MultiTask

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PatchTumorClassifyLitModule(LightningModule):
    """LightningModule for Brain Patch Tumor Classification."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        net: SwinUNETREnc128MultiTask,
        extra_kwargs: dict,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        extra_kwargs = self.hparams.extra_kwargs

        self.binary_classify_criterion = PatchClassifyBCELoss(
            mode="binary",
            pos_weights=extra_kwargs.pos_weights,
            patch_res=extra_kwargs.patch_sizes,
            scale_loss=1.0
        )

        self.multilabel_classify_criterion = PatchClassifyBCELoss(
            mode="multilabel",
            labels=extra_kwargs.labels,
            pos_weights=extra_kwargs.pos_weights,
            patch_res=extra_kwargs.patch_sizes,
            scale_loss=1.0
        )

        self.binary_classify_metric = BinaryPatchClassifyMetrics(
            metrics_names=extra_kwargs.metrics,
            summary_metrics_names=extra_kwargs.summary_metrics,
            patch_sizes=extra_kwargs.patch_sizes,
            threshold=extra_kwargs.patch_thresh,
            pr_curve_thresholds=extra_kwargs.pr_curve_thresholds
        )
        self.train_binary_classify_metric = self.binary_classify_metric.clone()
        self.val_binary_classify_metric = self.binary_classify_metric.clone()

        self.multilabel_classify_metric = MultiLabelPatchClassifyMetrics(
            metrics_names=extra_kwargs.metrics,
            summary_metrics_names=extra_kwargs.summary_metrics,
            labels=extra_kwargs.labels,
            patch_sizes=extra_kwargs.patch_sizes,
            threshold=extra_kwargs.patch_thresh,
            pr_curve_thresholds=extra_kwargs.pr_curve_thresholds
        )
        self.train_multilabel_classify_metric = self.multilabel_classify_metric.clone()
        self.val_multilabel_classify_metric = self.multilabel_classify_metric.clone()


    def forward(self, image: torch.Tensor):
        out = self.net(image)
        patch_embeddings, patch_preds = out["embeddings"], out["outs"]
        return patch_embeddings, patch_preds


    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        log.info("Started Training...")

    def training_step(self, batch: Any):
        # image: NxCxWxHxD (C=Channels)
        # mask: NxCxWxHxD  (C=Tumor subregions)
        image, target_patch_labels = (
            batch["image"],
            batch["patch_tumor_labels"],
        )
        _, patch_logits = self.forward(image)
        pred_binary, pred_multilabel = self._decompose_patch_labels(patch_logits)
        target_binary, target_multilabel = self._decompose_patch_labels(target_patch_labels)

        #calculate loss
        loss = {}
        binary_loss = self.binary_classify_criterion(pred_multilabel, target_multilabel)
        loss.update(binary_loss)
        multilabel_loss = self.multilabel_classify_criterion(pred_binary, target_binary)
        loss.update(multilabel_loss)
        loss['loss'] = (loss["binary_patch_classify_bce_loss"] + loss["multilabel_patch_classify_bce_loss"])*0.5

        #update metrics
        self.train_binary_classify_metric.update(pred_binary, target_binary)
        self.train_multilabel_classify_metric.update(pred_multilabel, target_multilabel)

        self.log_scores(
            loss, global_step=self.global_step, on_step=True, on_epoch=True, prog_bar=True, prefix="train/"
        )
        return loss["loss"]


    def on_train_epoch_end(self):
        train_metrics = {}
        train_binary_classify_metrics = self.train_binary_classify_metric.compute()['main_metrics']
        train_metrics.update(train_binary_classify_metrics)
        train_multilabel_classify_metrics = self.train_multilabel_classify_metric.compute()['main_metrics']
        train_metrics.update(train_multilabel_classify_metrics)
        
        self.log_scores(train_metrics, on_epoch=True, prog_bar=True, prefix='train/')


    def val_test_step(self, batch, mode="val"):
        # image: BxNx4xWxHxD (4 Image Channels,B=batch size,N=num of sliding windows in an image)
        # mask: BxNxCxWxHxD  (C= Tumor subregions Channels)
        image, target_patch_labels = (
            batch["image"],
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
        global_step: int=None,
        prefix="train/",
        on_epoch=None,
        on_step=None,
        prog_bar=False,
    ):  
        log_data = {}
        if global_step is not None:
            log_data.update({'global_step': global_step})
        log_data = log_data.update({f"{prefix}{k}": v for k, v in scores.items()})
        self.log_dict(log_data, on_epoch=on_epoch, on_step=on_step, prog_bar=prog_bar)


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
