import copy
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch.distributed as dist
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
    aggregate_timestep_quantile_losses,
    get_timestep_quantile_losses,
    mean_flat,
    params_to_state_dict,
    state_dict_to_params,
    update_ema,
    compute_subregions_pred_metrics,
)
from src.loss.brats_loss import BraTSLoss
from src.loss.denoising_loss import DenoisingLoss

from src.utils import RankedLogger
from typing import Callable, Any

from src.models.networks.unet.basic_unet_denoise import BasicUNetDenoise
from src.models.networks.unet.basic_unet import BasicUNetEncoder
from src.models.diffusion.build_diffusion import BuildDiffusion

from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, compute_hausdorff_distance

from src.models.diffusion.enums import *

log = RankedLogger(__name__, rank_zero_only=True)


class DenoisingDiffusionSimplNMbLitModule(LightningModule):
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
        embed_net: BasicUNetEncoder,
        diffusion: BuildDiffusion,
        sampler: ScheduleSampler,
        inferer: SlidingWindowInferer,
        extra_kwargs: dict,
        compile: bool,

    ) -> None:
        """Initialize a `DenoisingDiffusionLitModule`.

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
        self.embed_net: BasicUNetEncoder  = embed_net
        self.diffusion: SpacedDiffusion = diffusion.diffusion
        self.sample_diffusion: SpacedDiffusion = diffusion.sample_diffusion
        self.schedule_sampler: ScheduleSampler = sampler

        self.inferer: SlidingWindowInferer = inferer
        
        #self.train_loss = MeanMetric()
        self.criterion = BraTSLoss()
        self.denoising_criterion = DenoisingLoss(diffusion=self.diffusion)
        #self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True, ignore_empty=False)


    def on_load_checkpoint(self, checkpoint):
        self.ema_params = [
            state_dict_to_params(self.net, ema_state_dict)
            for ema_state_dict in checkpoint["ema_state_dicts"]
        ]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dicts"] = [
            params_to_state_dict(self.net, params) for params in self.ema_params
        ]


    def forward(self, image=None, x_start=None, x_t=None, t=None, denoise_out=None, pred_type=None):
        """Executes different functions of gaussian diffusion
            embeddings refer to image embeddings
        """
        if pred_type == "q_sample":
            noise = th.randn_like(x_start)
            x_t = self.diffusion.q_sample(x_start, t, noise=noise)
            return x_t, noise

        elif pred_type == "denoise_out":
            embeddings = self.embed_net(image)
            denoise_out = self.net(x_t, t=t, image=image, embeddings=embeddings)
            return denoise_out

        elif pred_type == "pred_xstart":
            #predict x_start from x_t
            if self.diffusion.model_mean_type == ModelMeanType.START_X:
                pred_xstart = denoise_out
            elif self.diffusion.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = self.diffusion._predict_xstart_from_eps(x_t=x_t, t=t, eps=denoise_out)
            else:
                raise NotImplementedError(self.diffusion.model_mean_type)
            return pred_xstart

        elif pred_type == "ddim_sample":
            embeddings = self.embed_net(image)
            B,C,W,H,D = image.shape
            sample_out = self.sample_diffusion.ddim_sample_loop(self.net, (B, self.hparams.extra_kwargs.number_targets, W, H,D), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["sample"]
            return sample_out


    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        log.info("Started Training...")

        # on_train_start, self.net.parameters() tensors are in device (cuda)
        # so, move ema_params tensors to same device (cuda) as net_params,
        # and then detach them so no gradient backprop on ema params
        ema_params_rates = []
        for rate, params in zip(self.ema_rate, self.ema_params):
            ema_params_rate = []
            for targ, src in zip(params, list(self.net.parameters())):
                targ = targ.to(src).detach()
                ema_params_rate.append(targ)
            ema_params_rates.append(ema_params_rate)
        self.ema_params = ema_params_rates


    def model_step(self, batch: Tuple[th.Tensor, Any]) -> th.Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return:
            - A tensor of losses.
        """
        image, mask, foreground = batch["image"], batch["mask"], batch["foreground"]
        mask = mask.float()
        x_start = (mask) * 2 - 1

        t, weights = self.schedule_sampler.sample(x_start)
        x_t, noise = self.forward(x_start=x_start, t=t, pred_type="q_sample")
        denoise_out = self.forward(x_t=x_t, t=t, image=image, pred_type="denoise_out")

        #denoising loss
        losses = self.denoising_criterion(model_output=denoise_out, x_start=x_start, x_t=x_t, t=t, noise=noise)
        deno_loss = (losses['loss'] * weights).mean()
        #update loss history (for importance sampling objective)
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, losses['loss'].detach())

        ##Compute losses b/w pred_xstart and mask_
        pred_xstart = self.forward(x_t=x_t,t=t,denoise_out=denoise_out, pred_type="pred_xstart") * foreground
        seg_losses = self.criterion(pred_xstart,mask)
        loss_dice = seg_losses['dice_loss']
        loss_bce = seg_losses['bce_loss']
        loss_mse = seg_losses['mse_loss']

        sum_loss = loss_dice + loss_bce

        if self.hparams.extra_kwargs.add_deno_loss:
            sum_loss += deno_loss
        if self.hparams.extra_kwargs.add_mse_loss:
            sum_loss += loss_mse

        seg_losses.update({'deno_loss':deno_loss,'loss':sum_loss})
        return seg_losses


    def training_step(self, batch: Tuple[th.Tensor, Any], batch_idx: int) -> th.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        all_losses = self.model_step(batch)
        self._log_scores(all_losses,on_step=True,on_epoch=True,prog_bar=True,prefix='train')

        # update ema
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, list(self.net.parameters()), rate=rate)

        # update and log metrics
        #self.train_loss(loss)

        self.log(
            "global_step", self.global_step, on_step=True, on_epoch=False, prog_bar=True
        )

        # return loss or backpropagation will fail
        return all_losses['loss']


    def on_train_batch_end(
        self, outputs: th.Tensor, batch: Any, batch_idx: int
    ) -> None:
        pass

    def val_test_step(self,batch):
        #image: Nx4xWxHxD (4 Image Channels)
        #mask: NxCxWxHxD  (C= Tumor subregions Channels)
        image, mask, foreground = batch["image"], batch["mask"], batch["foreground"]
        mask = mask.float()
        N,C,W,H,D = mask.shape
        subregions_names = self.trainer.datamodule.subregions_names
        assert len(subregions_names) == C

        #Evaluate on the entire image using sliding window inference
        logits = self.inferer(inputs=image,network=self.forward,pred_type="ddim_sample") * foreground

        #compute loss on raw logits
        seg_losses = self.criterion(logits, mask)
        seg_losses['loss'] = seg_losses["dice_loss"] + seg_losses["bce_loss"] + seg_losses["mse_loss"]
        self._log_scores(seg_losses,prefix='val',on_epoch=True,prog_bar=True)

        #compute scores on binary logits for every subregion
        logits = logits.sigmoid().gt(0.5)
        pred_scores = compute_subregions_pred_metrics(logits,mask,C,subregions_names)
        return logits,pred_scores
        #pred_scores = {f"val/{k}":v for k,v in pred_scores.items()}


    def validation_step(self, batch):
        _,pred_scores = self.val_test_step(batch)
        self._log_scores(pred_scores,prefix='val',on_epoch=True,prog_bar=True)


    def test_step(self, batch: Any, batch_idx: int):
        _,pred_scores = self.val_test_step(batch)
        self._log_scores(pred_scores,prefix='test',on_epoch=True,prog_bar=True)


    def predict_step(self, batch):
        datas, file_ids = batch
        images, foregrounds = datas["image"], datas["foreground"]
        logits = self.inferer(inputs=images,network=self.forward) * foregrounds
        logits = logits.sigmoid().gt(0.5)
        return logits,file_ids


    def _log_scores(self,scores:dict,prefix='train',on_epoch=False,on_step=False,prog_bar=False):
        scores = {f"{prefix}/{k}":v for k,v in scores.items()}
        self.log_dict(scores,on_epoch=on_epoch,on_step=on_step,prog_bar=prog_bar)


    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = th.compile(self.net)

        if stage == "fit":
            ema_rate = self.hparams.extra_kwargs.ema_rate
            self.ema_rate = (
                [ema_rate]
                if isinstance(ema_rate, float)
                else [float(x) for x in ema_rate.split(",")]
            )
            self.ema_params = [
                copy.deepcopy(list(self.net.parameters()))
                for _ in range(len(self.ema_rate))
            ]


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
