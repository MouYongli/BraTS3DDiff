from typing import Any, Dict, Tuple
import functools
import copy
import blobfile as bf
import numpy as np
import os

import torch as th
import lightning
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import torch.distributed as dist

from src.models.diffusion import gaussian_diffusion as gd
from src.models.diffusion.enums import ModelMeanType, ModelVarType, LossType
from src.models.networks.denoising_unet.unet import UNetModel
from src.models.diffusion.respace import SpacedDiffusion
from src.models.diffusion.enums import ModelMeanType, ModelVarType, LossType
from src.models.diffusion.resample import ScheduleSampler, LossAwareSampler

from src.models.utils.nn import update_ema, mean_flat
from src.models.utils import dist_util, logger
from src.models.utils.utils import state_dict_to_params, params_to_state_dict

from src.models.utils.create import create_gaussian_diffusion,create_model,create_named_schedule_sampler

class DenoisingDiffusionLitModule(LightningModule):
    """Example of a `LightningModule` for denosiing diffuiosn
    A `LightningModule` implements 8 key methods:

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
        kwargs:dict        
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

        logger.log("creating model and diffusion...")

        self.net: UNetModel = create_model(**net_cfg)
        self.diffusion: SpacedDiffusion = create_gaussian_diffusion(**diffusion_cfg)

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

        self.train_loss = MeanMetric()
        self.automatic_optimization = False


    def on_load_checkpoint(self, checkpoint):
        self.ema_params = [state_dict_to_params(self.net,ema_state_dict) for ema_state_dict in checkpoint['ema_state_dicts']]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dicts"] = [params_to_state_dict(self.net,params) for params in self.ema_params]


    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

        logger.log("Started Training...")
        #self._load_and_sync_parameters()
        #dist_util.sync_params(self.net.parameters())
        schedule_sampler = self.hparams.kwargs.schedule_sampler
        self.schedule_sampler: ScheduleSampler = create_named_schedule_sampler(schedule_sampler, self.diffusion)

        #on_train_start, self.net.parameters() tensors are in device (cuda)
        #so, move ema_params tensors to same device (cuda) as net_params, 
        # and then detach them so no gradient backprop on ema params
        ema_params_rates = []
        for rate, params in zip(self.ema_rate, self.ema_params):
            ema_params_rate = []
            for targ, src in zip(params, list(self.net.parameters())):
                targ = targ.to(src).detach()
                ema_params_rate.append(targ)

            #dist_util.sync_params(ema_params_rate)
            ema_params_rates.append(ema_params_rate)
        self.ema_params=ema_params_rates

        microbatch = self.hparams.kwargs.microbatch
        self.microbatch = microbatch if microbatch > 0 else self.batch_size
        self.global_batch = self.batch_size * dist.get_world_size()


    def forward(self, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.diffusion.loss_type == LossType.KL or self.diffusion.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self.diffusion._vb_terms_bpd(
                model=self.net,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.diffusion.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.diffusion.num_timesteps
        elif self.diffusion.loss_type == LossType.MSE or self.diffusion.loss_type == LossType.RESCALED_MSE:
            model_output = self.net(x_t, self.diffusion._scale_timesteps(t), **model_kwargs)

            if self.diffusion.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self.diffusion._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.diffusion.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.diffusion.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.diffusion.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.diffusion.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.diffusion.loss_type)

        return terms


    def model_step(
        self, batch: Tuple[th.Tensor, Any]
    ) -> th.Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
        """
        data, cond = batch

        for i in range(0, data.shape[0], self.microbatch):
            micro = data[i : i + self.microbatch]
            micro_cond = {
                k: v[i : i + self.microbatch]
                for k, v in cond.items()
            }
            #last_batch = (i + self.microbatch) >= data.shape[0]

            t, weights = self.schedule_sampler.sample(micro)

            losses = self.forward(micro,t,model_kwargs=micro_cond)

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            self.manual_backward(loss)

        return loss


    def training_step(
        self, batch: Tuple[th.Tensor, Any], batch_idx: int
    ) -> th.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        opt = self.optimizers()
        opt.zero_grad() #or __zero_grad()
        loss = self.model_step(batch)
        opt.step()
        #print('Before update',self.ema_params[0][0])
        self._update_ema()
        #print('After update',self.ema_params[0][0])

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("global_step", self.global_step, on_step=True, on_epoch=False, prog_bar=True)

        self.log_step()
        # return loss or backpropagation will fail
        return loss

    def on_train_batch_end(self, outputs: th.Tensor, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)

        if self.global_step % self.log_interval == 0:
            logger.dumpkvs()

    
    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, list(self.net.parameters()), rate=rate)

    def sample_loop_progressive(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        progress=False,
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


    def on_predict_start(self) -> None:
        #generated sampled images, and
        #their class labels
        self.all_images = []
        self.all_labels = []


    def predict_step(self, batch: Tuple[th.Tensor, Any]):

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
            use_ddim=use_ddim
        ):
            sample = sample_

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        self.all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_labels = None
        if self.hparams.net_cfg.class_cond:
            gathered_labels = [
                th.zeros_like(cond['y']) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, cond['y'])
            self.all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        logger.log(f"created {len(self.all_images) * self.batch_size} samples")
        return gathered_samples, gathered_labels


    def on_predict_end(self) -> None:
        num_samples = self.hparams.kwargs.num_samples
        class_cond = self.hparams.net_cfg.class_cond
        
        arr = np.concatenate(self.all_images, axis=0)
        arr = arr[: num_samples]
        if class_cond:
            label_arr = np.concatenate(self.all_labels, axis=0)
            label_arr = label_arr[: num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(self.hparams.kwargs.output_dir, f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

        dist.barrier()
        logger.log("sampling complete")


    def log_step(self):
        logger.logkv("step", self.global_step)
        logger.logkv("samples", (self.global_step+ 1) * self.global_batch)



    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = th.compile(self.net)

        self.use_ddp = isinstance(self.trainer.strategy, lightning.pytorch.strategies.ddp.DDPStrategy)


    def test_step(self, batch: Tuple[th.Tensor, Any], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass


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


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
            



if __name__ == "__main__":
    _ = DenoisingDiffusionLitModule(None, None, None, None)
