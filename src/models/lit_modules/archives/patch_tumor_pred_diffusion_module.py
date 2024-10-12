import copy
import os
from typing import Any, Dict, Tuple
import torch
import numpy as np

from lightning import LightningModule
from torchmetrics import MeanMetric
import torch as th
from src.models.diffusion import gaussian_diffusion as gd
from src.models.diffusion.enums import LossType, ModelMeanType, ModelVarType
from src.models.diffusion.noise_schedule import get_named_beta_schedule
from src.models.diffusion.respace import SpacedDiffusion, space_timesteps
from src.models.diffusion.timestep_sampler import (
    LossAwareSampler,
    LossSecondMomentResampler,
    ScheduleSampler,
    UniformSampler,
)
from src.models.utils.utils import (
    compute_subregions_pred_metrics,
    compute_uncertainty_based_fusion,
)
from src.loss.brats_loss import BraTSLoss
from src.loss.denoising_loss import DenoisingLoss

from src.utils import RankedLogger
from typing import Callable, Any

from src.models.networks.unet.basic_unet_denoise import BasicUNetDenoise
from src.models.networks.unet.basic_unet import BasicUNetEncoder
from src.models.diffusion.build_diffusion import BuildDiffusion

from monai.inferers.inferer import Inferer, SlidingWindowInferer
from src.inferer.sliding_window_infer import CustomSlidingWindowInferer
from monai.metrics import DiceMetric, compute_hausdorff_distance
from src.models.diffusion.enums import *
from src.utils.visualization import plot_mask
from src.models.networks.swinunetr.swinunetr_enc import SwinUNETREnc
from src.loss.patch_tumor_loss import PatchTumorLoss


log = RankedLogger(__name__, rank_zero_only=True)


class PatchTumorDiffusionLitModule(LightningModule):
    """Example of a `LightningModule` for denosiing diffuiosn A `LightningModule` implements 8 key
    methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: th.optim.Optimizer,
        scheduler: th.optim.lr_scheduler,
        net: BasicUNetDenoise,
        embed_net: SwinUNETREnc,
        diffusion: BuildDiffusion,
        sampler: ScheduleSampler,
        inferer: SlidingWindowInferer,
        extra_kwargs: dict,
        compile: bool,
    ) -> None:
        """Initialize a `PatchTumorDiffusionLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        log.info("Creating model and diffusion...")
        self.net: BasicUNetDenoise = net
        self.embed_net: SwinUNETREnc = embed_net
        self.diffusion: SpacedDiffusion = diffusion.diffusion
        self.sample_diffusion: SpacedDiffusion = diffusion.sample_diffusion
        self.schedule_sampler: ScheduleSampler = sampler

        self.inferer: SlidingWindowInferer = inferer

        # self.train_loss = MeanMetric()
        self.criterion = BraTSLoss()
        self.patch_criterion = PatchTumorLoss(
            mode="classify", patch_res=self.hparams.extra_kwargs.patch_sizes
        )
        self.denoising_criterion = DenoisingLoss(diffusion=self.diffusion)
        # self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True, ignore_empty=False)
        # self.automatic_optimization = False

    def on_load_checkpoint(self, checkpoint):
        pass

    def on_save_checkpoint(self, checkpoint):
        pass

    def forward(self, **kwargs):
        """Executes different functions of gaussian diffusion
        embeddings refer to image embeddings
        """

        def _q_sample(x_start, t):
            noise = th.randn_like(x_start)
            x_t = self.diffusion.q_sample(x_start, t, noise=noise)
            return x_t, noise

        def _get_patch_embeddings(image):
            out = self.embed_net(image, pred_mode="classify")
            patch_embeddings, patch_preds = out["embeddings"], out["labels"]
            return patch_embeddings, patch_preds

        def _denoise(x_t, t, patch_size, patch_embeddings):
            denoise_out = self.net(
                x_t, t=t, image=patch_embeddings, embeddings=None, patch_size=patch_size
            )
            return denoise_out

        def _pred_xstart(x_t, t, denoise_out):
            if self.diffusion.model_mean_type == ModelMeanType.START_X:
                pred_xstart = denoise_out
            elif self.diffusion.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = self.diffusion._predict_xstart_from_eps(
                    x_t=x_t, t=t, eps=denoise_out
                )
            else:
                raise NotImplementedError(self.diffusion.model_mean_type)
            return pred_xstart

        def _ddim_sample(image):
            embeddings = self.embed_net(image)
            B, C, W, H, D = image.shape
            sample_out = self.sample_diffusion.ddim_sample_loop(
                self.net,
                (B, self.hparams.extra_kwargs.number_targets, W, H, D),
                model_kwargs={"image": image, "embeddings": embeddings},
            )
            sample_out = sample_out["sample"]
            return sample_out

        def _ddim_sample_uncer_aware(image):
            # uncertainty fusion based frpm diffunet
            embeddings = self.embed_net(image)
            B, C, W, H, D = image.shape
            uncer_step = self.hparams.extra_kwargs.uncer_step
            sample_outputs = []
            for i in range(uncer_step):
                sample_outputs.append(
                    self.sample_diffusion.ddim_sample_loop(
                        self.net,
                        (B, self.hparams.extra_kwargs.number_targets, W, H, D),
                        model_kwargs={"image": image, "embeddings": embeddings},
                        viz_kwargs=None,
                    )
                )
            sample_return = compute_uncertainty_based_fusion(
                sample_outputs,
                (B, self.hparams.extra_kwargs.number_targets, W, H, D),
                uncer_step=uncer_step,
                num_sample_timesteps=self.sample_diffusion.num_timesteps,
            )
            return sample_return.to(image)

        pred_type = kwargs.get("pred_type")
        assert pred_type is not None

        if pred_type == "q_sample":
            return _q_sample(kwargs.get("x_start"), kwargs.get("t"))

        elif pred_type == "patch_embeddings":
            return _get_patch_embeddings(kwargs.get("image"))

        elif pred_type == "denoise_out":
            return _denoise(
                kwargs.get("x_t"),
                kwargs.get("t"),
                kwargs.get("patch_size"),
                kwargs.get("patch_embeddings"),
            )

        elif pred_type == "pred_xstart":
            # predict x_start from x_t
            return _pred_xstart(
                kwargs.get("x_t"), kwargs.get("t"), kwargs.get("denoise_out")
            )

        elif pred_type == "ddim_sample":
            return _ddim_sample(kwargs.get("image"))

        elif pred_type == "ddim_sample_uncer_aware":
            return _ddim_sample_uncer_aware(kwargs.get("image"))

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        log.info("Started Training...")

    def patch_model_step(self, image, patch_tumor_labels) -> th.Tensor:

        # get patch embeddings and labels from SwinT encoder
        patches_embeddings, patches_pred_labels = self.forward(
            image=image, pred_type="patch_embeddings"
        )
        patches_classify_loss = self.patch_criterion(
            patches_pred_labels, patch_tumor_labels
        )
        # self.manual_backward(patches_classify_loss['loss'])
        return patches_embeddings, patches_pred_labels, patches_classify_loss

    def diff_model_step(
        self, patches_embeddings, patches_pred_labels, mask, foreground
    ) -> th.Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return:
            - A tensor of losses.
        """

        B, C, W, H, D = mask.shape
        mask = mask.float()

        patch_sizes = self.trainer.datamodule.hparams.patch_sizes
        loss_dict = {"patch_loss": 0.0, "window_loss": 0.0}
        loss_dict.update(
            {
                f"{x}_{y}loss_res={k}": 0.0
                for x in ["patch", "window"]
                for y in ["dice_", "ce_", "deno_", ""]
                for k in patch_sizes
            }
        )

        pred_masks = []
        for i, patch_size in enumerate(patch_sizes):

            # reshape patch embeddings from 1-D features to 3-D
            patch_embeddings = patches_embeddings[patch_size]
            B, C_, W_, H_, D_ = patch_embeddings.shape
            patch_emb_size = self.hparams.extra_kwargs.patch_emb_size
            assert (
                (C_ == patch_emb_size**3)
                and (W_ == (W // patch_size))
                and (H_ == (H // patch_size))
                and (D_ == (D // patch_size))
            )
            patch_embeddings = patch_embeddings.view(
                B * W_ * H_ * D_, 1, patch_emb_size, patch_emb_size, patch_emb_size
            )

            # reshape mask to get mask patches
            mask_patch = mask.view(
                B * (W // patch_size) * (H // patch_size) * (D // patch_size),
                C,
                patch_size,
                patch_size,
                patch_size,
            )
            foreground_patch = foreground.view(
                B * (W // patch_size) * (H // patch_size) * (D // patch_size),
                1,
                patch_size,
                patch_size,
                patch_size,
            )

            # microbatch = self.hparams.extra_kwargs.microbatch

            pred_mask_patches = []
            n_microbatch = 0
            microbatch = B * (W // patch_size) * (H // patch_size) * (D // patch_size)
            for j in range(0, mask_patch.shape[0], microbatch):
                n_microbatch += 1

                patch_embeddings_ = patch_embeddings[j : j + microbatch]
                mask_patch_ = mask_patch[j : j + microbatch]
                foreground_patch_ = foreground_patch[j : j + microbatch]

                x_start = (mask_patch_) * 2 - 1

                t, weights = self.schedule_sampler.sample(x_start)
                x_t, noise = self.forward(x_start=x_start, t=t, pred_type="q_sample")
                denoise_out = self.forward(
                    x_t=x_t,
                    t=t,
                    patch_embeddings=patch_embeddings_,
                    patch_size=patch_size,
                    pred_type="denoise_out",
                )

                # denoising loss
                losses = self.denoising_criterion(
                    model_output=denoise_out, x_start=x_start, x_t=x_t, t=t, noise=noise
                )
                deno_loss = (losses["loss"] * weights).mean()
                # update loss history (for importance sampling objective)
                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                ##Compute seg losses b/w pred_xstart and mask_
                pred_xstart = (
                    self.forward(
                        x_t=x_t, t=t, denoise_out=denoise_out, pred_type="pred_xstart"
                    )
                    * foreground_patch_
                )
                pred_mask_patches.append(pred_xstart)

                patch_seg_losses = self.criterion(pred_xstart, mask_patch_)
                patch_loss_dice = patch_seg_losses["dice_loss"]
                patch_loss_bce = patch_seg_losses["bce_loss"]
                patch_loss = patch_loss_dice + patch_loss_bce + deno_loss
                # self.manual_backward(loss)

                loss_dict[f"patch_dice_loss_res={patch_size}"] += patch_loss_dice
                loss_dict[f"patch_ce_loss_res={patch_size}"] += patch_loss_bce
                loss_dict[f"patch_deno_loss_res={patch_size}"] += deno_loss
                loss_dict[f"patch_loss_res={patch_size}"] += patch_loss

            loss_dict[f"patch_dice_loss_res={patch_size}"] /= n_microbatch
            loss_dict[f"patch_ce_loss_res={patch_size}"] /= n_microbatch
            loss_dict[f"patch_deno_loss_res={patch_size}"] /= n_microbatch
            loss_dict[f"patch_loss_res={patch_size}"] /= n_microbatch
            loss_dict["patch_loss"] += loss_dict[f"patch_loss_res={patch_size}"]

            # Window level loss
            pred_mask = (
                torch.stack(pred_mask_patches, dim=0)
                .to(mask)
                .view(
                    B,
                    C,
                    (W // patch_size),
                    (H // patch_size),
                    (D // patch_size),
                    patch_size,
                    patch_size,
                    patch_size,
                )
            )
            patch_pred_labels = patches_pred_labels[patch_size].sigmoid().gt(0.5)
            patch_pred_labels = patch_pred_labels.view(
                B, 1, (W // patch_size), (H // patch_size), (D // patch_size), 1, 1, 1
            )
            patch_pred_labels = patch_pred_labels.expand(
                -1,
                C,
                (W // patch_size),
                (H // patch_size),
                (D // patch_size),
                patch_size,
                patch_size,
                patch_size,
            )
            pred_mask = pred_mask * patch_pred_labels
            pred_mask = pred_mask.view(B, C, W, H, D)
            pred_masks.append(pred_mask)

            window_seg_losses = self.criterion(pred_mask, mask)
            wloss_dice = window_seg_losses["dice_loss"]
            wloss_bce = window_seg_losses["bce_loss"]
            wloss = wloss_dice + wloss_bce
            # self.manual_backward(wloss)

            loss_dict[f"window_dice_loss_res={patch_size}"] = wloss_dice
            loss_dict[f"window_ce_loss_res={patch_size}"] = wloss_bce
            loss_dict[f"window_loss_res={patch_size}"] = wloss
            loss_dict["window_loss"] += loss_dict[f"window_loss_res={patch_size}"]

        # Mean of mask window predictions using all the patch_sizes
        mean_pred_mask = torch.mean(torch.stack(pred_masks), dim=0)
        window_seg_losses = self.criterion(mean_pred_mask, mask)
        wloss_dice = window_seg_losses["dice_loss"]
        wloss_bce = window_seg_losses["bce_loss"]
        wloss = wloss_dice + wloss_bce
        # self.manual_backward(wloss)

        loss_dict[f"window_dice_loss_res=mean"] = wloss_dice
        loss_dict[f"window_ce_loss_res=mean"] = wloss_bce
        loss_dict[f"window_loss_res=mean"] = wloss

        loss_dict["loss"] = (
            loss_dict["patch_loss"]
            + loss_dict["window_loss"]
            + loss_dict["window_loss_res=mean"]
        )
        return loss_dict

    def training_step(self, batch: Tuple[th.Tensor, Any], batch_idx: int) -> th.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        image, mask, foreground, patch_tumor_labels = (
            batch["image"],
            batch["mask"],
            batch["foreground"],
            batch["patch_tumor_labels"],
        )
        # opt = self.optimizers()
        # opt.zero_grad()  # or __zero_grad()
        patches_embeddings, patches_pred_labels, patches_classify_loss = (
            self.patch_model_step(image, patch_tumor_labels)
        )
        patch_classify_loss = patches_classify_loss.pop("loss")
        self.log(
            f"train/patch_classify_loss",
            patch_classify_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self._log_scores(
            patches_classify_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            prefix="train",
        )
        # opt.step()

        # opt.zero_grad()  # or __zero_grad()
        loss_dict = self.diff_model_step(
            patches_embeddings, patches_pred_labels, mask, foreground
        )
        diff_loss = loss_dict.pop("loss")
        self.log(
            f"train/diff_loss", diff_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self._log_scores(
            loss_dict, on_step=True, on_epoch=True, prog_bar=True, prefix="train"
        )

        # opt.step()
        loss = patch_classify_loss + diff_loss

        self.log(f"train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # update and log metrics
        # self.train_loss(loss)
        self.log(
            "global_step", self.global_step, on_step=True, on_epoch=False, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_batch_end(
        self, outputs: th.Tensor, batch: Any, batch_idx: int
    ) -> None:
        pass

    def val_test_step(self, batch, mode="val"):
        # image: Nx4xWxHxD (4 Image Channels)
        # mask: NxCxWxHxD  (C= Tumor subregions Channels)
        image, mask, foreground = batch["image"], batch["mask"], batch["foreground"]
        mask = mask.float()
        N, C, W, H, D = mask.shape
        subregions_names = self.trainer.datamodule.subregions_names
        im_channels = self.trainer.datamodule.im_channels
        assert len(subregions_names) == C

        # Evaluate on the entire image using sliding window inference
        if self.inferer.viz:
            logits = (
                self.inferer(
                    inputs=image,
                    network=self.forward,
                    pred_type=self.hparams.extra_kwargs.pred_type,
                    gt_mask=mask,
                    subregions=subregions_names,
                    im_channels=im_channels,
                )
                * foreground
            )
        else:
            logits = (
                self.inferer(
                    inputs=image,
                    network=self.forward,
                    pred_type=self.hparams.extra_kwargs.pred_type,
                )
                * foreground
            )

        # compute loss on raw logits
        seg_losses = self.criterion(logits, mask)
        seg_losses["loss"] = seg_losses["dice_loss"] + seg_losses["bce_loss"]
        self._log_scores(seg_losses, prefix=mode, on_epoch=True, prog_bar=True)
        self.log(
            f"{mode}/loss",
            seg_losses["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # compute scores on binary logits for every subregion
        logits = logits.sigmoid().gt(0.5)
        pred_scores = compute_subregions_pred_metrics(logits, mask, C, subregions_names)
        self._log_scores(pred_scores, prefix=mode, on_epoch=True, prog_bar=True)
        return logits, pred_scores
        # pred_scores = {f"val/{k}":v for k,v in pred_scores.items()}

    def validation_step(self, batch):
        return self.val_test_step(batch, mode="val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.val_test_step(batch, mode="test")

    def predict_step(self, batch):
        datas, file_ids = batch
        images, foregrounds = datas["image"], datas["foreground"]
        logits = (
            self.inferer(inputs=images, network=self.forward, pred_type="ddim_sample")
            * foregrounds
        )
        logits = logits.sigmoid().gt(0.5)
        return logits, file_ids

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

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = th.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = DenoisingDiffusionSimplNMbLitModule(None, None, None, None)
