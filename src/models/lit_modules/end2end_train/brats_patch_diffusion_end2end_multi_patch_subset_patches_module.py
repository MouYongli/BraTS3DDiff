import copy
import os
import time
from typing import Any, Dict, Tuple
import json
import numpy as np
import torch
from lightning import LightningModule
import wandb
from src.models.diffusion.enums import ModelMeanType
from src.models.diffusion.timestep_sampler import (
    LossAwareSampler,
    ScheduleSampler,
)
from src.models.diffusion.respace import SpacedDiffusion
from src.models.diffusion.build_diffusion import BuildDiffusion

from src.loss.seg_loss import MultiResPatchSegLoss
from src.loss.denoising_loss import DenoisingLoss
from src.loss.patch_classify_loss import MultiResPatchClassifyLoss

from src.metrics.multi_res_metrics import MultiResSegmentMetrics, MultiResPatchClassifyMetrics

from src.models.networks.patch_unet.patch_unet_denoise import PatchDenoiseUNet
from src.models.networks.patch_unet.patch_enc import PatchUpsample, PatchUNetEncoder
from src.models.networks.swinunetr.swinunetr_enc_1x1_conv import SwinUNETREnc128

from monai.inferers.inferer import SlidingWindowInferer

from src.utils.model_utils import (
    compute_uncertainty_based_fusion,
    window2patches,
    add_background_batch,
    sample_patch_indices,
    get_vals_from_idxs,
    ravel_tuple_index,
    stable_divide,
    patches2window,
    plot_grad_flow,
    compute_grad_norm,
    aggregate_common_x_and_2x_patches_mul
    
)
from src.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


class BraTSPatchTumorDiffusionLitModule(LightningModule):
    """

    ```
    Patch-based Diffusion with Prior Patch Tumor Classification
    During training, all patches in an image are noised using different timesteps

        > Single-Stage Approach (All losses combined and backprop on total loss)

    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        patchify_net: SwinUNETREnc128,
        patch_up_net: PatchUpsample,
        patch_emb_net: PatchUNetEncoder,
        patch_denoise_net: PatchDenoiseUNet,
        diffusion: BuildDiffusion,
        sampler: ScheduleSampler,
        inferer: SlidingWindowInferer,
        extra_kwargs: dict,
        compile: bool,
    ) -> None:
        """Initialize a `BraTSPatchTumorDiffusionLitModule`.

        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param patch_denoise_net: The denoising model
        :param patchify_net: The SwinT encoder for patchifying the image
        :param patch_emb_net: Network for upsampling SwinT patch embeddings,
                                and computing UNet Enc features on those embeddings
        :param diffusion: Module for building Gaussian Diffusion
        :param sampler: Timestep sampler
        :param inferer: Sliding Window Inferer for performing inference
        :param extra_kwargs: Extra Keyword arguments

        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        log.info("Creating model and diffusion...")

        self.patchify_net: SwinUNETREnc128 = patchify_net
        self.patch_up_net: PatchUpsample = patch_up_net # and up net
        self.patch_emb_net: PatchUNetEncoder = patch_emb_net
        self.patch_denoise_net: PatchDenoiseUNet = patch_denoise_net

        self.diffusion: SpacedDiffusion = diffusion.diffusion
        self.sample_diffusion: SpacedDiffusion = diffusion.sample_diffusion
        self.schedule_sampler: ScheduleSampler = sampler

        self.inferer: SlidingWindowInferer = inferer

        self.patch_sizes = self.hparams.extra_kwargs.patch_sizes
        self.patch_emb_sizes = self.hparams.extra_kwargs.patch_emb_sizes

        self.patch_classify_criterion = MultiResPatchClassifyLoss(
            mode="classify", patch_res=self.patch_sizes, scale_loss=1.0
        )
        self.denoising_criterion = DenoisingLoss(diffusion=self.diffusion)
        
        self.segment_criterion = MultiResPatchSegLoss(patch_res=self.patch_sizes, incl_mean=True,
                                    scale_loss=1./3, dice_batch=True, prefixes=['seg', 'masked_seg'])

        self.patch_classify_metric = MultiResPatchClassifyMetrics(patch_sizes=self.patch_sizes,
                                                                  sigmoid=True, thresh=self.hparams.extra_kwargs.patch_thresh)

        self.segment_metric = MultiResSegmentMetrics(patch_sizes=self.patch_sizes, incl_mean=True,
                                                     channels=self.hparams.extra_kwargs.subregions_names,
                                                     sigmoid=False, thresh=0.5)

        # self.automatic_optimization = False

    def on_load_checkpoint(self, checkpoint):
        pass

    def on_save_checkpoint(self, checkpoint):
        pass

    def forward(self, **kwargs):
        """Executes different functions of gaussian diffusion"""

        def _patchify_image(image):
            # image -> patch tokens
            out = self.patchify_net(image)
            patch_embeddings, patch_preds = out["embeddings"], out["labels"]
            return patch_embeddings, patch_preds

        def _up_and_embed(patch_embeddings, patch_size):
            # upscale patch_embeddings to match patch_size, and
            # compute embeddings for the upsampled representations
            patch = self.patch_up_net(patch_embeddings, patch_size)
            embeddings = self.patch_emb_net(patch)
            return patch, embeddings

        def _q_sample(x_start, t):
            noise = torch.randn_like(x_start)
            x_t = self.diffusion.q_sample(x_start, t, noise=noise)
            return x_t, noise

        def _denoise(x_t, t, patch, embeddings):
            denoise_out = self.patch_denoise_net(
                x_t, t=t, image=patch, embeddings=embeddings
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

        def _ddim_sample(mask_patch_shape, patch, embeddings):
            sample_out = self.sample_diffusion.ddim_sample_loop(
                self.patch_denoise_net,
                mask_patch_shape,
                model_kwargs={
                    "image": patch,
                    "embeddings": embeddings                
                },
            )
            sample_out = sample_out["sample"]
            return sample_out

        def _ddim_sample_uncer_aware(mask_patch_shape, patch, embeddings):
            # uncertainty fusion based from diffunet
            uncer_step = self.hparams.extra_kwargs.uncer_step
            sample_outputs = []
            for i in range(uncer_step):
                sample_outputs.append(
                    self.sample_diffusion.ddim_sample_loop(
                        self.patch_denoise_net,
                        mask_patch_shape,
                        model_kwargs={
                            "image": patch,
                            "embeddings": embeddings,
                        },
                        viz_kwargs=None,
                    )
                )

            sample_return = compute_uncertainty_based_fusion(
                sample_outputs,
                mask_patch_shape,
                uncer_step=uncer_step,
                num_sample_timesteps=self.sample_diffusion.num_timesteps,
            )

            return sample_return.to(patch)

        pred_type = kwargs.get("pred_type")
        assert pred_type is not None

        if pred_type == "patchify":
            return _patchify_image(kwargs.get("image"))

        elif pred_type == "up_and_embed":
            return _up_and_embed(
                kwargs.get("patch_embeddings"), kwargs.get("patch_size")
            )

        elif pred_type == "q_sample":
            return _q_sample(kwargs.get("x_start"), kwargs.get("t"))

        elif pred_type == "denoise_out":
            return _denoise(
                kwargs.get("x_t"),
                kwargs.get("t"),
                kwargs.get("patch"),
                kwargs.get("embeddings"),
            )

        elif pred_type == "pred_xstart":
            # predict x_start from x_t
            return _pred_xstart(
                kwargs.get("x_t"), kwargs.get("t"), kwargs.get("denoise_out")
            )

        elif pred_type == "ddim_sample":
            return _ddim_sample(
                kwargs.get("mask_patch_shape"),
                kwargs.get("patch"),
                kwargs.get("embeddings"),
            )

        elif pred_type == "ddim_sample_uncer_aware":
            return _ddim_sample_uncer_aware(
                kwargs.get("mask_patch_shape"),
                kwargs.get("patch"),
                kwargs.get("embeddings"),
            )

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        log.info("Started Training...")

    def on_train_start(self):
        #define necessary variables for patch sampling 
        self.start_eps = [float(start_eps) for start_eps in self.hparams.extra_kwargs.start_eps]
        self.end_eps = [float(end_eps) for end_eps in self.hparams.extra_kwargs.end_eps]
        self.num_epochs_use = [epochs-1 for epochs in self.hparams.extra_kwargs.num_epochs_use]
        self.num_patch_samples = self.hparams.extra_kwargs.num_samples
        self.curr_eps = [None]*len(self.patch_sizes)
        assert len(self.patch_sizes) == len(self.patch_emb_sizes) == len(self.start_eps) == \
                len(self.end_eps) == len(self.num_epochs_use) == len(self.num_patch_samples)

        wandb.watch(self, log='all', log_freq=20, log_graph=True)


    def on_train_epoch_start(self):
        for i, (start_eps, end_eps, num_epochs_use) in enumerate(zip(self.start_eps, self.end_eps, self.num_epochs_use)):
            if self.trainer.current_epoch <= num_epochs_use:
                #linear eps scheduling: reduce eps over epochs linearly
                curr_eps = start_eps - ((start_eps-end_eps)*(self.trainer.current_epoch/num_epochs_use))
                self.curr_eps[i] = curr_eps
            self.log(f'patch_sample_res={self.patch_sizes[i]}/curr_eps', self.curr_eps[i],  on_epoch=True, prog_bar=True)


    def _get_sampled_patches_stats(self, patch_size, patch_tumor_vol_fracs, patch_tumor_labels, sampled_patch_indices_nd):
        sampled_patch_tumor_vol_fracs = get_vals_from_idxs(patch_tumor_vol_fracs, sampled_patch_indices_nd)
        sampled_patch_tumor_labels = get_vals_from_idxs(patch_tumor_labels, sampled_patch_indices_nd)
        total_all_patch_tumor_vol_fracs = patch_tumor_vol_fracs.sum()
        total_sampled_patch_tumor_vol_fracs = sampled_patch_tumor_vol_fracs.sum()
        frac_tumor_covered = stable_divide(total_sampled_patch_tumor_vol_fracs, total_all_patch_tumor_vol_fracs)

        num_all_tumor_patches = patch_tumor_labels.sum()
        num_sampled_tumor_patches = sampled_patch_tumor_labels.sum()
        frac_sampled_tumor_patches = num_sampled_tumor_patches/num_all_tumor_patches
        tumor_to_nontumor_ratio_all_patches = stable_divide(num_all_tumor_patches, patch_tumor_labels.numel()-num_all_tumor_patches)
        num_patch_samples = sampled_patch_tumor_labels.shape[0]
        tumor_to_nontumor_ratio_sampled_patches = num_sampled_tumor_patches/(num_patch_samples-num_sampled_tumor_patches)

        sampled_patch_stats = dict(
            total_all_patch_tumor_vol_fracs=total_all_patch_tumor_vol_fracs,
            total_sampled_patch_tumor_vol_fracs=total_sampled_patch_tumor_vol_fracs,
            frac_tumor_covered=frac_tumor_covered,
            num_all_tumor_patches=num_all_tumor_patches,
            num_sampled_tumor_patches=num_sampled_tumor_patches,
            frac_sampled_tumor_patches=frac_sampled_tumor_patches,
            tumor_to_nontumor_ratio_all_patches=tumor_to_nontumor_ratio_all_patches,
            tumor_to_nontumor_ratio_sampled_patches=tumor_to_nontumor_ratio_sampled_patches
        )

        self.log_scores(scores=sampled_patch_stats, prefix=f'patch_sample_res={patch_size}', on_step=True, on_epoch=True, prog_bar=True)



    def model_step(self, batch) -> torch.Tensor:
        """Perform a single model step on a batch of data.
            1. Compute patch embeddings and pred patch tumor labels using SwinT encoder
            2. Compute patch classification loss between pred patch tumor labels & GT patch tumor labels
            3. Upsample patch embeddings to match patch size
            4. x_start: whole mask, x_t: noise applied to whole mask
            5. x_t_patch: break down noisy mask into patches
            6. Denoise noisy patches x_t_patch with corresponding patch embeddings as condition
            7. Reshape denoised patches to get denoised whole mask, apply denoising loss
            8. Compute image wise segmentation loss between whole denoised seg mask and GT seg mask
            9. Mask predicted seg mask using pred patch tumor labels and, again compute segmentation loss
            10. Sum all losses

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return:
            - A tensor of losses.
        """

        image, mask, patches_tumor_vol_fracs, patches_tumor_labels = (
            batch["image"],
            batch["mask"],
            batch["patch_tumor_vol_fracs"],
            batch["patch_tumor_labels"]

        )

        B, C, W, H, D = mask.shape
        loss_dict = {'deno_loss': 0}

        true_patch_seg_masks = {}
        pred_patch_seg_masks = {}
        masked_patch_pred_seg_masks = {}

        sampled_patches_indices = {}

        # get patch embeddings and pred patch labels from SwinT encoder
        # compute patch classify loss
        patches_embeddings, patches_pred_labels = self.forward(
            image=image, pred_type="patchify"
        )
        patches_classify_loss_dict = self.patch_classify_criterion(patches_pred_labels, patches_tumor_labels)
        loss_dict.update(patches_classify_loss_dict)

        self.segment_criterion._init_loss()

        for i, patch_size in enumerate(self.patch_sizes):
            patch_emb_size = self.patch_emb_sizes[i]
            curr_eps = self.curr_eps[i]
            num_patch_samples = self.num_patch_samples[i]
            #self.log(f'patch_sample_res={self.patch_sizes[i]}/eps', curr_eps,  on_epoch=True, prog_bar=True)

            patch_embeddings = patches_embeddings[str(patch_size)] #(B,C_,W_,H_,D_)
            patch_pred_labels = patches_pred_labels[str(patch_size)].sigmoid()

            #randomly sample patches based on their gt tumor vol fracs
            sampled_patch_indices_flat,  sampled_patch_indices_nd = sample_patch_indices(
                patches_tumor_vol_fracs[str(patch_size)], eps=curr_eps, num_samples=num_patch_samples
            )
            sampled_patches_indices[str(patch_size)] = sampled_patch_indices_flat

            patch_embeddings = get_vals_from_idxs(patch_embeddings, sampled_patch_indices_nd).view(
                num_patch_samples, 1, patch_emb_size, patch_emb_size, patch_emb_size
            )
            
            patch_pred_labels = get_vals_from_idxs(patch_pred_labels, sampled_patch_indices_nd).view(
                num_patch_samples, 1, 1, 1, 1
            )

            #for debugging: get sampled patches stats
            self._get_sampled_patches_stats(
                patch_size, patches_tumor_vol_fracs[str(patch_size)], 
                patches_tumor_labels[str(patch_size)], sampled_patch_indices_nd
            )

            # upsample the patch embeddings from patch_emb_size to match the patch_size resolutions,
            # and embed the upsampled patches
            patch, embeddings = self.forward(
                patch_embeddings=patch_embeddings,
                patch_size=patch_size,
                pred_type="up_and_embed",
            )
            assert patch.shape[2:] == (
                patch_size,
                patch_size,
                patch_size,
            )

            # get mask patches from the whole mask and add all mask patches in the batch dimension
            # x_start (B,C,W,H,D) -> mask_patch (B*W_*H_*D_,C,patch_size,patch_size,patch_size)
            mask_patch = window2patches(mask.float(), patch_size)
            mask_patch = mask_patch[sampled_patch_indices_flat]
            true_patch_seg_masks[str(patch_size)] = mask_patch

            # mask_: x_start
            x_start_patch = (mask_patch) * 2 - 1

            # apply noise to x_start_patch
            t_patch, weights_patch = self.schedule_sampler.sample(x_start_patch)
            x_t_patch, noise_patch = self.forward(x_start=x_start_patch, t=t_patch, pred_type="q_sample")

            # denoise the noised mask patches
            denoise_out_patch = self.forward(
                x_t=x_t_patch,
                t=t_patch,
                patch=patch,
                embeddings=embeddings,
                pred_type="denoise_out",
            )

            # denoising loss on patches
            deno_loss = self.denoising_criterion(
                model_output=denoise_out_patch, x_start=x_start_patch, x_t=x_t_patch, t=t_patch, noise=noise_patch, weights=weights_patch
            )
            loss_dict[f"deno_loss_res={patch_size}"] = deno_loss
            loss_dict[f"deno_loss"] += deno_loss
            # update deno loss history (for importance sampling objective)
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t_patch, deno_loss.detach()
                )

            # Get x_start_patch (B*W_*H_,D_,C,patch_size,patch_size,patch_size)
            pred_mask_patch = self.forward(
                x_t=x_t_patch,
                t=t_patch,
                denoise_out=denoise_out_patch,
                pred_type="pred_xstart",
            )

            pred_mask_patch = pred_mask_patch.sigmoid()
            pred_patch_seg_masks[str(patch_size)] = pred_mask_patch

            # Scale the predicted segmentation mask using the patch pred tumor labels and then compute seg loss
            masked_pred_mask_patch = patch_pred_labels * pred_mask_patch
            masked_patch_pred_seg_masks[str(patch_size)] = masked_pred_mask_patch

            #Compute losses
            self.segment_criterion(pred_mask_patch, mask_patch, patch_res=patch_size, prefix='seg')
            self.segment_criterion(masked_pred_mask_patch, mask_patch, patch_res=patch_size, prefix='masked_seg')

        #aggregate overlapping 16 and 32 patches
        [mean_pred_mask_patch, masked_mean_pred_mask_patch], common_patch_indices = aggregate_common_x_and_2x_patches_mul(
            [pred_patch_seg_masks['16'], masked_patch_pred_seg_masks['16']], 
            sampled_patches_indices['16'],
            patches_tumor_labels['16'].shape,
            [pred_patch_seg_masks['32'], masked_patch_pred_seg_masks['32']], 
            sampled_patches_indices['32'],
            patches_tumor_labels['32'].shape,
            agg='mean'
        )

        #Compute losses
        self.segment_criterion(mean_pred_mask_patch, true_patch_seg_masks['16'][common_patch_indices[:,0]], patch_res='mean', prefix='seg')
        self.segment_criterion(masked_mean_pred_mask_patch, true_patch_seg_masks['16'][common_patch_indices[:,0]], patch_res='mean', prefix='masked_seg')

        self.segment_criterion._scale_loss()
        loss_dict.update(self.segment_criterion.loss_dict)
        self.segment_criterion._reset_loss()

        loss_dict["loss"] = (
            loss_dict["patch_classify_loss"] #bce_patch_16+bce_patch_32
            + loss_dict["deno_loss"] #deno_mse_patch_16+deno_mse_patch_32
            + loss_dict["seg_loss"] #seg_loss_patch_16+seg_loss_patch_32+seg_loss_patch_mean (seg_loss_=(dice+bce))
            + loss_dict["masked_seg_loss"] #same as seg_loss
        )
        return loss_dict


    def training_step(self, batch: Tuple[torch.Tensor, Any], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        loss_dict = self.model_step(batch)
        self.log_scores(
            loss_dict, on_step=True, on_epoch=True, prog_bar=True, prefix="train"
        )

        self.log(
            "global_step", self.global_step, on_step=True, on_epoch=False, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss_dict["loss"]


    def on_after_backward(self):
        #plot gradient statistics
        if self.global_step % self.hparams.extra_kwargs.plot_grad_flow_freq == 0:
            models = {'patchify_net':self.patchify_net,'patch_up_net':self.patch_up_net,
                        'patch_emb_net':self.patch_emb_net,'patch_denoise_net':self.patch_denoise_net}
            log_dict = {}
            for model_name, model in models.items():
                fig, fig_norm = plot_grad_flow(model)
                log_dict[f'gradient_stats/grad_stats_{model_name}'] = wandb.Image(fig)
                log_dict[f'gradient_stats/grad_norm_{model_name}'] = wandb.Image(fig_norm)
                log_dict[f'gradient_stats/total_grad_norm_{model_name}'] = compute_grad_norm(model)        
            wandb.log(log_dict)



    def predict_seg_mask(self, image, ret_patch_labels=False):
        """
        predict the segmentation mask for a given image window
            1. Compute patch embeddings and patch tumor labels using SwinT encoder
            2. Find tumor patch indices and get tumor patch embeddings
            3. Upsample tumor patch embeddings to match patch_size
            4. Init zero tensor of pred_mask
            5. ddim sample tumor mask patches corresponding to the tumor patch embeddings
            6. Fill in pred_mask zero tensor with the sampled tumor mask patches, at the tumor patch indices
            7. Perform previous steps for multiple patch sizes and average the output pred_masks

        Args:
            image: (B,C,W,H,D)
            ret_patch_labels: If True, returns predicted patch tumor labels (used for validation)
        Returns:
            out: dict( key: "res=patch_size", val: pred_mask (B,C,W,H,D) )
        """

        start_time = time.time()
        B, _, W, H, D = image.shape
        C = self.hparams.extra_kwargs.num_targets
        mask_shape = (B, C, W, H, D)

        # get patch embeddings and pred patch labels from SwinT encoder
        patches_embeddings, patches_pred_labels = self.forward(
            image=image, pred_type="patchify"
        )

        pred_masks = {}

        for i, patch_size in enumerate(self.patch_sizes):

            patch_emb_size = self.patch_emb_sizes[i]
            C_ = patch_emb_size**3
            # patch locations
            W_, H_, D_ = (W // patch_size, H // patch_size, D // patch_size)

            # filter tumor patch locations using patches_pred_labels and generate masks only for these patch locations
            # binarize patch tumor predictions
            patch_pred_labels = (
                patches_pred_labels[str(patch_size)]
                .sigmoid()
                .gt(self.hparams.extra_kwargs.patch_thresh)
            ) 

            assert patch_pred_labels.shape == (B, 1, W_, H_, D_)

            # get patch_embeddings for the current patch_size
            patch_embeddings = patches_embeddings[str(patch_size)]
            assert patch_embeddings.shape == (B, C_, W_, H_, D_)

            #filter patch_embeddings using the predicted patch labels to get the tumor_patch_embeddings
            tumor_patch_indices = patch_pred_labels.nonzero(as_tuple=True)

            num_tumor_patches = tumor_patch_indices[0].shape[0]

            # create a zero tensor for batch of predicted masks
            pred_mask = torch.zeros(B*W_*H_*D_, C, patch_size, patch_size, patch_size).to(image)

            if num_tumor_patches > 0:
                #get tumor patch embeddings
                patch_embeddings = get_vals_from_idxs(patch_embeddings, tumor_patch_indices).view( 
                num_tumor_patches, 1, patch_emb_size, patch_emb_size, patch_emb_size
            )

                # upsample the tumor patch embeddings from patch_emb_size to match the patch_size resolutions and embed the upsampled patches
                patch, embeddings = self.forward(
                    patch_embeddings=patch_embeddings,
                    patch_size=patch_size,
                    pred_type="up_and_embed",
                )
                assert patch.shape[2:] == (
                    patch_size,
                    patch_size,
                    patch_size,
                )

                # sample the tumor mask patches using ddim with tumor_patch_embeddings as condition
                mask_patch_shape = (
                    num_tumor_patches,
                    C,
                    patch_size,
                    patch_size,
                    patch_size,
                )
                pred_tumor_mask_patch = self.forward(
                    pred_type=self.hparams.extra_kwargs.sampling_type,
                    mask_patch_shape=mask_patch_shape,
                    patch=patch,
                    embeddings=embeddings,
                )
                # fill in the zero tensor pred_mask created before, at the tumor patch locations
                #  with the corresponding predicted mask patches
                tumor_patch_indices_flat = ravel_tuple_index(tumor_patch_indices, patch_pred_labels.shape)
                pred_mask[tumor_patch_indices_flat] = pred_tumor_mask_patch.sigmoid()

            pred_mask = patches2window(pred_mask, win_size=(W, H, D))
            pred_masks[str(patch_size)] = pred_mask

        # Mean of mask window predictions using all the patch sizes
        mean_pred_mask = torch.stack(list(pred_masks.values())).mean(dim=0)
        pred_masks['mean'] = mean_pred_mask

        if ret_patch_labels:
            return pred_masks, patches_pred_labels
        else:
            return pred_masks


    def validation_step(self, batch):
        # validation is performed on random crops of whole image
        start_time = time.time()
        image, mask, patch_tumor_labels = (
            batch["image"],
            batch["mask"],
            batch["patch_tumor_labels"],
        )
        pred_masks, pred_patch_tumor_labels = self.predict_seg_mask(
            image, ret_patch_labels=True
        )
        #compute losses
        val_losses = {}
        ##patch classify loss
        patch_classify_losses = self.patch_classify_criterion(
            pred_patch_tumor_labels, patch_tumor_labels
        )
        val_losses.update(patch_classify_losses)

        ##seg loss
        self.segment_criterion._init_loss(prefixes=['seg'])
        for key in pred_masks.keys():
            self.segment_criterion(pred_masks[key], mask, patch_res=key, prefix='seg')
        self.segment_criterion._scale_loss(prefixes=['seg'])
        val_losses.update(self.segment_criterion.loss_dict)
        self.segment_criterion._reset_loss()

        val_losses['loss'] = val_losses['patch_classify_loss'] + val_losses['seg_loss']
        self.log_scores(val_losses, prefix="val", on_epoch=True, prog_bar=True)

        #compute metrics
        self.patch_classify_metric(preds=pred_patch_tumor_labels, trues=patch_tumor_labels)
        self.segment_metric(preds=pred_masks, trues=mask)

        dur = time.time() - start_time
        log.info(f"One val step with {image.shape} images took {dur:0.4f} secs")


    def on_validation_epoch_end(self):
        val_metrics = {}
        patch_classify_metrics, confmats = self.patch_classify_metric.compute_metrics()
        log.info(f"Confmats: {json.dumps(confmats)}")
        val_metrics.update(patch_classify_metrics)
        seg_metrics = self.segment_metric.compute_metrics()
        val_metrics.update(seg_metrics)
        self.log_scores(val_metrics, prefix="val", on_epoch=True, prog_bar=True)
        self.log(
            f"val/dice",
            val_metrics["seg_dice_mean"],
            on_epoch=True,
            prog_bar=True,
        )


    def test_step(self, batch):
        # test is performed on whole image using sliding windows
        start_time = time.time()
        data, file_id, orig_img_shape = batch
        image, mask, fg_start, fg_end = data["image"], data["mask"], data['foreground_start_coord'], data['foreground_end_coord']

        pred_seg_masks = self.inferer(inputs=image, network=self.predict_seg_mask)

        ##seg loss
        self.segment_criterion._init_loss(prefixes=['seg'])
        for key in pred_seg_masks.keys():
            self.segment_criterion(pred_seg_masks[key], mask, patch_res=key, prefix='seg')
        self.segment_criterion._scale_loss(prefixes=['seg'])
        self.log_scores(self.segment_criterion.loss_dict, prefix="test", on_epoch=True, prog_bar=True)
        self.segment_criterion._reset_loss()

        self.segment_metric(preds=pred_seg_masks, trues=mask)

        for key in pred_seg_masks.keys():
            pred_seg_mask = pred_seg_masks[key].gt(0.5)
            pred_seg_masks[key] = add_background_batch(pred_seg_mask, orig_img_shape, fg_start, fg_end)

        dur = time.time() - start_time
        log.info(f"One test step with {image.shape} images took {dur:0.4f} secs")
        return pred_seg_masks, file_id


    def on_test_epoch_end(self):
        seg_metrics = self.segment_metric.compute_metrics()
        self.log_scores(seg_metrics, prefix="test", on_epoch=True, prog_bar=True)
        self.log(
            f"test/dice",
            seg_metrics["seg_dice_mean"],
            on_epoch=True,
            prog_bar=True,
        )



    def predict_step(self, batch):
        start_time = time.time()

        data, file_id, orig_mask_shape = batch
        image, fg_start, fg_end = data["image"], data['foreground_start_coord'], data['foreground_end_coord']
        pred_seg_masks = self.inferer(inputs=image, network=self.predict_seg_mask)

        for key in pred_seg_masks.keys():
            pred_seg_mask = pred_seg_masks[key].gt(0.5)
            pred_seg_masks[key] = add_background_batch(pred_seg_mask, orig_mask_shape, fg_start, fg_end)

        dur = time.time() - start_time
        log.info(f"One predict step with {image.shape} images took {dur:0.4f} secs")

        return pred_seg_masks, file_id


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

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            #self.patch_denoise_net = torch.compile(self.patch_denoise_net)
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
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = BraTSPatchTumorDiffusionLitModule(None, None, None, None)
