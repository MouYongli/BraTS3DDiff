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

from src.guided_diffusion import gaussian_diffusion as gd
from src.guided_diffusion.respace import SpacedDiffusion, space_timesteps
from src.guided_diffusion.unet import SuperResModel, UNetModel, EncoderUNetModel
from src.guided_diffusion.resample import ScheduleSampler

from dataclasses import dataclass, Field
from .model_utils import create_gaussian_diffusion, create_model, zero_grad, state_dict_to_params
from src.guided_diffusion.resample import create_named_schedule_sampler, LossAwareSampler
from src.guided_diffusion.nn import update_ema
from src.guided_diffusion import logger, dist_util




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
        self.log_interval = self.kwargs.get('log_interval', 10)
        self.resume_step = 0
        self.checkpoint = self.kwargs.get('checkpoint', "")
        self.batch_size = self.kwargs.get('batch_size')

        logger.log("creating model and diffusion...")

        self.net: UNetModel = create_model(**net_cfg)
        self.diffusion: SpacedDiffusion = create_gaussian_diffusion(**diffusion_cfg)

        self.net_params = list(self.net.parameters())

        self._load_parameters()
 
        self.train_loss = MeanMetric()
        self.automatic_optimization = False

 
    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.net_params)

        main_checkpoint = self.checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=self.device,
                )
                ema_params = state_dict_to_params(self.net,state_dict)

        #dist_util.sync_params(ema_params)
        return ema_params


    def _get_optimizer_state(self):
        main_checkpoint = self.checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=self.device,
            )
            return state_dict
            #opt = self.optimizers().optimizer
            #self.opt.load_state_dict(state_dict)


    def _load_parameters(self):
        if self.checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {self.checkpoint}...")
                self.net.load_state_dict(
                    dist_util.load_state_dict(
                        self.checkpoint, map_location=self.device,
                    )
                )

        #dist_util.sync_params(self.net.parameters())
        

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        pass
    
    
    def on_predict_start(self) -> None:
        #generated sampled images, and
        #their class labels
        self.all_images = []
        self.all_labels = []


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

    
    def predict_step(self, batch: Tuple[th.Tensor, Any]):
        
        noise, cond = batch
        
        use_ddim = self.hparams.kwargs.use_ddim
        sample_fn = (
            self.diffusion.p_sample_loop if not use_ddim else self.diffusion.ddim_sample_loop
        )
        im_size = self.hparams.net_cfg.image_size
        sample = sample_fn(
            self.net,
            (noise.shape[0], 3, im_size, im_size),
            noise=noise,
            clip_denoised=self.kwargs.clip_denoised,
            model_kwargs=cond,
        )
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
    
    
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

        logger.log("Started Training...")

        #self._load_and_sync_parameters()
        dist_util.sync_params(self.net.parameters())

        schedule_sampler = self.kwargs.get('schedule_sampler','loss-second-moment')
        self.schedule_sampler: ScheduleSampler = create_named_schedule_sampler(schedule_sampler, self.diffusion)

        ema_rate = self.kwargs.get('ema_rate', 0.999)
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )



        if self.resume_step:
            #self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.net_params)
                for _ in range(len(self.ema_rate))
            ]

        #on_train_start, self.net_params tensors are in device (cuda)
        #so, move ema_params tensors to same device (cuda) as net_params, 
        # and then detach them so no gradient backprop on ema params
        ema_params_rates = []
        for rate, params in zip(self.ema_rate, self.ema_params):
            ema_params_rate = []
            for targ, src in zip(params, self.net_params):
                targ = targ.to(src).detach()
                ema_params_rate.append(targ)

            dist_util.sync_params(ema_params_rate)
            ema_params_rates.append(ema_params_rate)

        self.ema_params=ema_params_rates
        
        
        microbatch = self.kwargs.get('microbatch', -1)
        self.microbatch = microbatch if microbatch > 0 else self.batch_size
        self.global_batch = self.batch_size * dist.get_world_size()



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
            last_batch = (i + self.microbatch) >= data.shape[0]
            #t, weights = self.schedule_sampler.sample(micro.shape[0],self.device)
            t, weights = self.schedule_sampler.sample(micro)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.net,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.trainer.model.no_sync():
                    losses = compute_losses()

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
        self.log_step()
        # return loss or backpropagation will fail
        return loss

    def on_train_batch_end(self, outputs: th.Tensor, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)

        if self.global_step % self.log_interval == 0:
            logger.dumpkvs()

    
    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.net_params, rate=rate)


    def log_step(self):
        logger.logkv("step", self.trainer.global_step + self.resume_step)
        logger.logkv("samples", (self.trainer.global_step + self.resume_step + 1) * self.global_batch)



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
        optimizer = self.hparams.optimizer(params=self.net_params)
        if self.resume_step:
            optimizer.load_state_dict(self._get_optimizer_state())

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
            
def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


if __name__ == "__main__":
    _ = DenoisingDiffusionLitModule(None, None, None, None)
