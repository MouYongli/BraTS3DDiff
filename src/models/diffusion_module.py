import copy
import os
from typing import Any, Dict, Tuple

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from lightning import LightningModule
from torchmetrics import MeanMetric
import torch
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
from src.models.networks.denoising_unet import UNetModel
from src.models.utils.utils import (
    aggregate_timestep_quantile_losses,
    get_timestep_quantile_losses,
    mean_flat,
    params_to_state_dict,
    state_dict_to_params,
    update_ema,
)
from src.loss.brats_loss import BraTSLoss
from src.loss.denoising_loss import DenoisingLoss

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def create_model(
    image_size: int = 64,
    num_channels: int = 128,
    num_res_blocks: int = 2,
    channel_mult: str = "",
    learn_sigma: bool = False,
    class_cond: bool = False,
    num_classes: int = None,
    use_checkpoint: bool = False,
    attention_resolutions: str = "16",
    num_heads: int = 1,
    num_head_channels: int = -1,
    num_heads_upsample: int = -1,
    use_scale_shift_norm: bool = False,
    dropout: int = 0,
    resblock_updown: bool = False,
    use_fp16: bool = False,
    use_new_attention_order: bool = False,
) -> UNetModel:
    if class_cond:
        assert num_classes is not None
    else:
        assert num_classes is None

    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(num_classes if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_gaussian_diffusion(
    steps: int = 1000,
    learn_sigma: bool = False,
    sigma_small: bool = False,
    noise_schedule: str = "linear",
    use_kl: bool = False,
    predict_xstart: bool = False,
    rescale_timesteps: bool = False,
    rescale_learned_sigmas: bool = False,
    timestep_respacing: str = "",
) -> SpacedDiffusion:
    betas = get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X
        ),
        model_var_type=(
            (ModelVarType.FIXED_LARGE if not sigma_small else ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def create_named_schedule_sampler(name, diffusion):
    """Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class DenoisingDiffusionLitModule(LightningModule):
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
        compile: bool,
        net_cfg: dict,
        diffusion_cfg: dict,
        kwargs: dict,
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

        self.kwargs = kwargs
        self.log_interval = self.hparams.kwargs.log_interval
        self.batch_size = self.hparams.kwargs.batch_size

        log.info("Creating model and diffusion...")

        self.net: UNetModel = create_model(**net_cfg)
        self.diffusion: SpacedDiffusion = create_gaussian_diffusion(**diffusion_cfg)

        #self.train_loss = MeanMetric()
        self.criterion = BraTSLoss()
        self.denoising_criterion = DenoisingLoss(diffusion=self.diffusion)
        self.automatic_optimization = False

    def on_load_checkpoint(self, checkpoint):
        self.ema_params = [
            state_dict_to_params(self.net, ema_state_dict)
            for ema_state_dict in checkpoint["ema_state_dicts"]
        ]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dicts"] = [
            params_to_state_dict(self.net, params) for params in self.ema_params
        ]


    def forward(self, pred_type=None, image=None, embeddings=None, x_start=None, x_t=None, t=None, model_out=None):
        """Executes different functions of gaussian diffusion
            embeddings refer to image embeddings
        """
        if pred_type == "q_sample":
            noise = torch.randn_like(x_start)
            x_t = self.diffusion.q_sample(x_start, t, noise=noise)
            return x_t, noise

        elif pred_type == "model_out":
            model_out = self.net(x_t, t=self.diffusion._scale_timesteps(t), image=image, embeddings=embeddings)
            return model_out

        elif pred_type == "pred_xstart":
            #predict x_start from x_t
            if not model_out:
                model_out = self.net(x_t, t=self.diffusion._scale_timesteps(t), image=image, embeddings=embeddings)

            if self.diffusion.model_mean_type == ModelMeanType.PREVIOUS_X:
                pred_xstart = self.diffusion._predict_xstart_from_xprev(x_t=x_t, t=t, xprev=model_out)
            elif self.diffusion.model_mean_type == ModelMeanType.START_X:
                pred_xstart = model_out
            elif self.diffusion.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = self.diffusion._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_out)
            else:
                raise NotImplementedError(self.diffusion.model_mean_type)
            return pred_xstart

        elif pred_type == "ddim_sample":
            #embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, roi_size[0], roi_size[0], roi_size[0]), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out




    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

        log.info("Started Training...")
        schedule_sampler = self.hparams.kwargs.schedule_sampler
        self.schedule_sampler: ScheduleSampler = create_named_schedule_sampler(
            schedule_sampler, self.diffusion
        )

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

        microbatch = self.hparams.kwargs.microbatch
        self.microbatch = microbatch if microbatch > 0 else self.batch_size

    def model_step(self, batch: Tuple[th.Tensor, Any]) -> th.Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return:
            - A tensor of losses.
        """
        image, mask, foreground = batch["image"], batch["mask"], batch["foreground"]

        qt_losses_dict = {}
        for i in range(0, image.shape[0], self.microbatch):
            image_ = image[i : i + self.microbatch]
            mask_ = mask[i : i + self.microbatch]
            foreground_ = foreground[i : i + self.microbatch]

            x_start = (mask_) * 2 - 1

            t, weights = self.schedule_sampler.sample(x_start)
            x_t, noise = self.forward(x=x_start, pred_type="q_sample")
            model_out = self.forward(x_t=x_t, t=t, image=image_, pred_type="model_out")
            losses = self.denoising_criterion(model_out=model_out, x_start=x_start, x_t=x_t, t=t, noise=noise, loss_type="general_denoising_loss")

            # compute quantile losses for timesteps
            qt_losses_dict = get_timestep_quantile_losses(
                t, weights, losses, self.diffusion.num_timesteps, qt_losses_dict
            )
            #update loss history (for importance sampling objective)
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses['loss'].detach())

            #denoising loss
            deno_loss = (losses['loss'] * weights).mean()

            ##Compute losses b/w pred_xstart and mask_
            pred_xstart = self.forward(x_t=x_t,step=t,model_out=model_out, pred_type="pred_xstart") * foreground_
            #pred_xstart = torch.sigmoid(pred_xstart)
            #pred_xstart_mse = self.mse(pred_xstart, mask_)
            #Q1: is computing mse on pred_x_start and x_start necessary for all the model mean configs?
            ##eg., if the model predicts the noise, the simple_mse_loss computes the mse b/w true and predicted noise
            ###but, is computing mse on x_start and interpolated x_start (using x_t and pred noise: q(x_t|x_start)) also necessary for all the model mean configs?
            ##########################################################
            #Q2: Should the foreground mask and torch.sigmoid() be first applied to pred_x_start 
            ## before applying the mse b/w pred_x_start and x_start (mask_),

            seg_losses = self.criterion(pred_xstart,mask_)
            loss_dice = seg_losses['dice']
            loss_bce = seg_losses['bce']
            loss_mse = seg_losses['mse']

            loss = deno_loss + loss_dice + loss_bce + loss_mse

            self.manual_backward(loss)

        qt_losses_dict = aggregate_timestep_quantile_losses(qt_losses_dict)
        self.log_dict(qt_losses_dict, on_step=True, prog_bar=True)

        return loss

    def training_step(self, batch: Tuple[th.Tensor, Any], batch_idx: int) -> th.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        opt = self.optimizers()
        opt.zero_grad()  # or __zero_grad()
        loss = self.model_step(batch)
        opt.step()

        # update ema
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, list(self.net.parameters()), rate=rate)

        # update and log metrics
        #self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(
            "global_step", self.global_step, on_step=True, on_epoch=False, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_batch_end(
        self, outputs: th.Tensor, batch: Any, batch_idx: int
    ) -> None:
        if self.global_step % self.log_interval == 0:
            pass
            # logger.dumpkvs()

    def on_predict_start(self) -> None:
        # generated sampled images, and
        # their class labels
        self.all_images = []
        self.all_labels = []

    def sample_loop_progressive(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=None,
        use_ddim=False,
    ):
        if not use_ddim:
            assert not eta
        if use_ddim and (not eta):
            eta = 0.0

        """
        Use DDP/IM to sample from the model and yield intermediate samples from
        each timestep of DDP/IM.

        Same usage as p_sample_loop_progressive().
        """
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=self.device)
        indices = list(range(self.diffusion.num_timesteps))[::-1]

        # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm

        indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=self.device)
            with th.no_grad():
                if use_ddim:
                    out = self.diffusion.ddim_sample(
                        self.net,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                    )
                else:
                    out = self.diffusion.p_sample(
                        self.net,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                    )
                img = out["sample"]
                yield img

    def predict_step(self, batch: Tuple[th.Tensor, Any]):
        """Perform a single sampling step on a batch of noise samples and target labels
            Generates a batch of images corresponding to target labels, and
            Gathers generated images from all the processes

        :param batch: A batch of data (a tuple) containing the input tensor of noise and target
            labels.
        :return: Does not return
        """
        # batch consists of noise, and class condition
        # converts noise to image of class condition

        use_ddim = self.hparams.kwargs.use_ddim
        eta = self.hparams.kwargs.eta
        im_size = self.hparams.net_cfg.image_size

        if not use_ddim:
            assert not eta
        if use_ddim and (not eta):
            eta = 0.0

        noise, cond = batch

        sample = None
        for sample_ in self.sample_loop_progressive(
            (noise.shape[0], 3, im_size, im_size),
            noise=noise,
            clip_denoised=self.kwargs.clip_denoised,
            model_kwargs=cond,
            eta=eta,
            use_ddim=use_ddim,
        ):
            sample = sample_

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        if self.trainer.world_size > 1:
            gathered_samples = self.all_gather(sample)
        else:
            gathered_samples = sample.unsqueeze(0)

        self.all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_labels = None
        if self.hparams.net_cfg.class_cond:
            if self.trainer.world_size > 1:
                gathered_labels = self.all_gather(cond["y"])
            else:
                gathered_labels = cond["y"].unsqueeze(0)
            self.all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        log.info(f"created {len(self.all_images)*noise.shape[0]} samples")

    def on_predict_end(self) -> None:
        num_samples = self.hparams.kwargs.num_samples
        class_cond = self.hparams.net_cfg.class_cond

        arr = np.concatenate(self.all_images, axis=0)
        arr = arr[:num_samples]
        if class_cond:
            label_arr = np.concatenate(self.all_labels, axis=0)
            label_arr = label_arr[:num_samples]
        if self.global_rank == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(
                self.hparams.kwargs.output_dir, f"samples_{shape_str}.npz"
            )
            log.info(f"saving to {out_path}")
            if class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

        if dist.is_available():
            dist.barrier()

        log.info("sampling complete")

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
            ema_rate = self.hparams.kwargs.ema_rate
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
    _ = DenoisingDiffusionLitModule(None, None, None, None)
