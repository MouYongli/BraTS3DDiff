import copy
import os
from typing import Any, Dict, Tuple
import torch
import numpy as np
import time

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

from src.models.networks.unet.basic_unet_denoise import PatchDenoiseUNet
from src.models.networks.unet.basic_unet import PatchUNetEncoder
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
        denoise_net: PatchDenoiseUNet,
        patchify_net: SwinUNETREnc,
        patch_emb_net: PatchUNetEncoder,
        diffusion: BuildDiffusion,
        sampler: ScheduleSampler,
        inferer: SlidingWindowInferer,
        extra_kwargs: dict,
        compile: bool,

    ) -> None:
        """Initialize a `PatchTumorDiffusionLitModule`.

        :param denoise_net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        log.info("Creating model and diffusion...")

        self.denoise_net: PatchDenoiseUNet = denoise_net
        self.patchify_net: SwinUNETREnc  = patchify_net
        self.patch_emb_net: PatchUNetEncoder = patch_emb_net

        self.diffusion: SpacedDiffusion = diffusion.diffusion
        self.sample_diffusion: SpacedDiffusion = diffusion.sample_diffusion
        self.schedule_sampler: ScheduleSampler = sampler

        self.inferer: SlidingWindowInferer = inferer

        #self.train_loss = MeanMetric()
        self.criterion = BraTSLoss()
        self.patch_criterion = PatchTumorLoss(mode='classify',patch_res=self.hparams.extra_kwargs.patch_sizes,avg=True)
        self.denoising_criterion = DenoisingLoss(diffusion=self.diffusion)

        #self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True, ignore_empty=False)
        #self.automatic_optimization = False

    def on_load_checkpoint(self, checkpoint):
        pass

    def on_save_checkpoint(self, checkpoint):
        pass


    def forward(self, **kwargs):
        """Executes different functions of gaussian diffusion
            embeddings refer to image embeddings
        """

        def _patchify_image(image):
            #image -> patch tokens
            out = self.patchify_net(image,pred_mode='classify')
            patch_embeddings, patch_preds = out['embeddings'],out['labels']
            return patch_embeddings, patch_preds

        def _up_and_embed(patch_embeddings, patch_size):
            #upscale patch_embeddings to match patch_size, and
            #compute embeddings for the upsampled representations
            patch, embeddings = self.patch_emb_net(patch_embeddings, patch_size)
            return patch, embeddings

        def _q_sample(x_start,t):
            noise = th.randn_like(x_start)
            x_t = self.diffusion.q_sample(x_start, t, noise=noise)
            return x_t, noise

        def _denoise(x_t, t, patch, embeddings, patch_size):
            denoise_out = self.denoise_net(x_t, t=t, image=patch, embeddings=embeddings, patch_size=patch_size)
            return denoise_out

        def _pred_xstart(x_t,t,denoise_out):
            if self.diffusion.model_mean_type == ModelMeanType.START_X:
                pred_xstart = denoise_out
            elif self.diffusion.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = self.diffusion._predict_xstart_from_eps(x_t=x_t, t=t, eps=denoise_out)
            else:
                raise NotImplementedError(self.diffusion.model_mean_type)
            return pred_xstart

        def _ddim_sample(mask_patch_shape, patch, embeddings, patch_size):
            sample_out = self.sample_diffusion.ddim_sample_loop(self.denoise_net, mask_patch_shape, \
                                                    model_kwargs={"image": patch, "embeddings": embeddings, "patch_size": patch_size})
            sample_out = sample_out["sample"]
            return sample_out

        def _ddim_sample_uncer_aware(mask_patch_shape, patch, embeddings, patch_size):
            #uncertainty fusion based from diffunet
            uncer_step = self.hparams.extra_kwargs.uncer_step
            sample_outputs = []
            for i in range(uncer_step):
                sample_outputs.append(self.sample_diffusion.ddim_sample_loop(self.denoise_net, mask_patch_shape, \
                                                                model_kwargs={"image": patch, "embeddings": embeddings, "patch_size": patch_size},viz_kwargs=None))

            sample_return = compute_uncertainty_based_fusion(sample_outputs, mask_patch_shape, \
                                                uncer_step=uncer_step, num_sample_timesteps=self.sample_diffusion.num_timesteps)

            return sample_return.to(patch)


        pred_type = kwargs.get('pred_type')
        assert pred_type is not None

        if pred_type == 'patchify':
            return _patchify_image(kwargs.get('image'))

        elif pred_type == 'up_and_embed':
            return _up_and_embed(kwargs.get('patch_embeddings'), kwargs.get('patch_size'))

        elif pred_type == "q_sample":
            return _q_sample(kwargs.get('x_start'),kwargs.get('t'))

        elif pred_type == "denoise_out":
            return _denoise(kwargs.get('x_t'),kwargs.get('t'),kwargs.get('patch'),kwargs.get('embeddings'),kwargs.get('patch_size'))

        elif pred_type == "pred_xstart":
            #predict x_start from x_t
            return _pred_xstart(kwargs.get('x_t'),kwargs.get('t'),kwargs.get('denoise_out'))

        elif pred_type == "ddim_sample":
            return _ddim_sample(kwargs.get('mask_patch_shape'),kwargs.get('patch'),kwargs.get('embeddings'),kwargs.get('patch_size'))

        elif pred_type == "ddim_sample_uncer_aware":
            return _ddim_sample_uncer_aware(kwargs.get('mask_patch_shape'),kwargs.get('patch'),kwargs.get('embeddings'),kwargs.get('patch_size'))



    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        log.info("Started Training...")


    def patch_model_step(self, image, patch_tumor_labels) -> th.Tensor:

        #get patch embeddings and labels from SwinT encoder
        patches_embeddings, patches_pred_labels = self.forward(image=image,pred_type='patch_embeddings')
        patches_classify_loss = self.patch_criterion(patches_pred_labels, patch_tumor_labels)
        #self.manual_backward(patches_classify_loss['loss'])
        return patches_embeddings, patches_pred_labels, patches_classify_loss


    def model_step(self, batch) -> th.Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return:
            - A tensor of losses.
        """

        image, mask, foreground, patch_tumor_labels = batch["image"], batch["mask"], batch["foreground"], batch["patch_tumor_labels"]

        B,C,W,H,D = mask.shape
        mask = mask.float()

        #get patch embeddings and pred patch labels from SwinT encoder
        #compute patch classify loss
        patches_embeddings, patches_pred_labels = self.forward(image=image,pred_type='patchify')
        patches_classify_loss_dict = self.patch_criterion(patches_pred_labels, patch_tumor_labels)

        loss_dict = {'deno_loss':0.0, 'seg_loss':0.0, 'masked_seg_loss':0.0}
        loss_dict.update(patches_classify_loss_dict)

        #loss_dict.update({f"{x}_{y}loss_res={k}":0.0 for x in ['patch_seg', 'window_seg'] for y in ['dice_','ce_','deno_',''] for k in patch_sizes})
        #patch_sizes = self.trainer.datamodule.hparams.patch_sizes
        patch_sizes = self.hparams.extra_kwargs.patch_sizes
        assert len(patch_sizes) == 2
        pred_seg_masks = []
        masked_pred_seg_masks = []

        for i, patch_size in enumerate(patch_sizes):

            patch_embeddings = patches_embeddings[patch_size]
            B,C_,W_,H_,D_ = patch_embeddings.shape
            patch_emb_size = self.hparams.extra_kwargs.patch_emb_size
            assert (C_==patch_emb_size**3) and (W_==(W//patch_size)) and (H_==(H//patch_size)) and (D_==(D//patch_size))
            W_,H_,D_ = (W//patch_size,H//patch_size,D//patch_size)

            #reshape patch embeddings from 1-D features to 3-D features, and add all patch locations to the batch dimension
            #patch_embeddings: B,C_,W_,H_,D_ --> B*W_*H_,D_ x 1 x patch_emb_size x patch_emb_size x patch_emb_size
            patch_embeddings = patch_embeddings.permute(0,2,3,4,1).contiguous().view(-1,1,patch_emb_size,patch_emb_size,patch_emb_size)
            assert patch_embeddings.shape  == (B*W_*H_*D_, 1, patch_emb_size, patch_emb_size, patch_emb_size)

            #upsample the patch embeddings from patch_emb_size to match the patch_size resolutions and embed the upsampled patches
            patch, embeddings = self.forward(patch_embeddings=patch_embeddings, patch_size=patch_size, pred_type="up_and_embed")
            assert patch.shape  == (B*W_*H_*D_, image.shape[1], patch_size, patch_size, patch_size)

            #mask_patch = mask.view(B,C,W_,patch_size,H_,patch_size,D_,patch_size)
            #mask_patch = mask_patch.permute(0,2,4,6,1,3,5,7).contiguous().view(-1,C,patch_size,patch_size,patch_size)

            #reshape foreground to get foreground patches and add all foreground patches in the batch dimension
            #(B,1,W,H,D) -> (B*W_*H_*D_,1,patch_size,patch_size,patch_size)
            #foreground_patch = foreground.view(B,1,W_,patch_size,H_,patch_size,D_,patch_size)
            #foreground_patch = foreground_patch.permute(0,2,4,6,1,3,5,7).contiguous().view(-1,1,patch_size,patch_size,patch_size)

            # mask_: x_start
            x_start = (mask) * 2 - 1
            #reshape mask to get mask patches and add all mask patches in the batch dimension
            #mask (B,C,W,H,D) -> (B*W_*H_*D_,C,patch_size,patch_size,patch_size)

            # apply noise to mask (x_start)
            t, weights = self.schedule_sampler.sample(x_start)
            x_t, noise = self.forward(x_start=x_start, t=t, pred_type="q_sample")

            #get noised mask patches from the entire noisy mask x_t and add all mask patches in the batch dimension
            #all mask patches have the same noise level
            #mask (B,C,W,H,D) -> noised_mask_patch (B*W_*H_*D_,C,patch_size,patch_size,patch_size)
            x_t_patch = x_t.view(B,C,W_,patch_size,H_,patch_size,D_,patch_size)
            x_t_patch = x_t_patch.permute(0,2,4,6,1,3,5,7).contiguous().view(-1,C,patch_size,patch_size,patch_size)

            #repeat t and weight values for all the patches
            t_patch = t.repeat_interleave(W_*H_*D_)

            #denoise the noised mask patches
            denoise_out_patch = self.forward(x_t=x_t_patch, t=t_patch, patch=patch, embeddings=embeddings, patch_size=patch_size, pred_type="denoise_out")

            # Reshape denoise_out_patch to get the window level denoise_out
            # (B*W_*H_,D_,C,patch_size,patch_size,patch_size) -> (B,C,W_,patch_size,H_,patch_size,D_,patch_size) -> (B,C,W,H,D)
            denoise_out = denoise_out_patch.view(B, W_, H_, D_, C, patch_size, patch_size, patch_size).permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(B, C, W, H, D)

            # denoising loss on the entire window
            losses = self.denoising_criterion(model_output=denoise_out, x_start=x_start, x_t=x_t, t=t, noise=noise)

            deno_loss = (losses['loss'] * weights).mean()
            loss_dict[f"deno_loss_res={patch_size}"] = deno_loss
            loss_dict[f"deno_loss"] += deno_loss

            #update deno loss history (for importance sampling objective)
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses['loss'].detach())


            ###Compute Segmentation Loss
            #Get predicted x_start ie the predicted mask patches (B*W_*H_,D_,C,patch_size,patch_size,patch_size)
            pred_mask_patch = self.forward(x_t=x_t_patch, t=t_patch, denoise_out=denoise_out_patch, pred_type="pred_xstart")
            pred_mask = pred_mask_patch.view(B, W_, H_, D_, C, patch_size, patch_size, patch_size).permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(B, C, W, H, D)
            pred_seg_masks.append(pred_mask)

            ##Compute seg losses b/w predicted mask and true mask
            seg_losses = self.criterion(pred_mask * foreground, mask)
            loss_dict[f"seg_dice_loss_res={patch_size}"] = seg_losses['dice_loss']
            loss_dict[f"seg_ce_loss_res={patch_size}"] = seg_losses['bce_loss']
            loss_dict[f"seg_loss_res={patch_size}"] = seg_losses['dice_loss'] + seg_losses['bce_loss']
            loss_dict['seg_loss'] += loss_dict[f"seg_loss_res={patch_size}"]


            #Mask the predicted segmentation mask using the tumor patch labels and then compute Window level seg loss
            #mask the non-tumor patches of pred_mask so that it gets a penalty if the predicted patch labels are wrong

            #binarize pred labels of patches (tumor/non-tumor labels)
            patch_pred_labels = patches_pred_labels[patch_size].sigmoid().gt(0.5)
            #expand shape of patch_pred_labels so that it matches the shape of pred_mask (so that non-tumor patches of pred_mask can be masked with )
            #patch_pred_labels (B,1,W_,H_,D_) -> (B,1,W_,1,H_,1,D_,1) -> B,C,W_,P,H_,P,D_,P -> B,C,W,H,D
            patch_pred_labels = patch_pred_labels.view(B, 1, W_, 1, H_, 1, D_, 1).expand(B, C, W_, patch_size, H_, patch_size, D_, patch_size).contiguous().view(B, C, W, H, D)

            masked_pred_mask = patch_pred_labels * pred_mask
            masked_pred_seg_masks.append(masked_pred_mask)

            masked_seg_losses = self.criterion(masked_pred_mask * foreground, mask)
            loss_dict[f"masked_seg_dice_loss_res={patch_size}"] = masked_seg_losses['dice_loss']
            loss_dict[f"masked_seg_ce_loss_res={patch_size}"] = masked_seg_losses['bce_loss']
            loss_dict[f"masked_seg_loss_res={patch_size}"] = masked_seg_losses['dice_loss'] + masked_seg_losses['bce_loss']
            loss_dict['masked_seg_loss'] += loss_dict[f"masked_seg_loss_res={patch_size}"]


        #scale loss by no. of patch resolutions
        #loss_dict['seg_loss'] /= len(patch_sizes)
        #loss_dict['masked_seg_loss'] /= len(patch_sizes)
        loss_dict[f"deno_loss"] /= len(patch_sizes)

        #Mean of mask window predictions using all the patch_sizes
        mean_pred_mask = torch.mean(torch.stack(pred_seg_masks), dim=0)
        seg_losses = self.criterion(mean_pred_mask * foreground, mask)
        loss_dict[f"seg_dice_loss_res=mean"] = seg_losses['dice_loss']
        loss_dict[f"seg_ce_loss_res=mean"] = seg_losses['bce_loss']
        loss_dict[f"seg_loss_res=mean"] = seg_losses['dice_loss'] + seg_losses['bce_loss']

        loss_dict['seg_loss'] += loss_dict[f"seg_loss_res=mean"]
        loss_dict['seg_loss'] /= (len(patch_sizes) + 1)

        # (MASKED) Mean of mask window predictions using all the patch_sizes
        masked_mean_pred_mask = torch.mean(torch.stack(masked_pred_seg_masks), dim=0)
        masked_seg_losses = self.criterion(masked_mean_pred_mask * foreground, mask)
        loss_dict[f"masked_seg_dice_loss_res=mean"] = masked_seg_losses['dice_loss']
        loss_dict[f"masked_seg_ce_loss_res=mean"] = masked_seg_losses['bce_loss']
        loss_dict[f"masked_seg_loss_res=mean"] = masked_seg_losses['dice_loss'] + masked_seg_losses['bce_loss']

        loss_dict['masked_seg_loss'] += loss_dict[f"masked_seg_loss_res=mean"]
        loss_dict['masked_seg_loss'] /= (len(patch_sizes) + 1)

        '''
        #Consistency between pred seg masks predicted using different patch sizes
        consistency_losses = self.criterion(pred_seg_masks[0], pred_seg_masks[1].sigmoid())
        loss_dict[f"seg_consistency_loss"] = consistency_losses['bce_loss']

        #Consistency between masked pred seg masks predicted using different patch sizes
        masked_consistency_losses = self.criterion(masked_pred_seg_masks[0], masked_pred_seg_masks[1].sigmoid())
        loss_dict[f"masked_seg_consistency_loss"] = masked_consistency_losses['bce_loss']
        '''

        loss_dict['loss'] = loss_dict['patch_classify_loss'] + loss_dict[f"deno_loss"] + loss_dict['seg_loss'] + loss_dict['masked_seg_loss']
        return loss_dict


    def training_step(self, batch: Tuple[th.Tensor, Any], batch_idx: int) -> th.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        loss_dict = self.model_step(batch)
        self._log_scores(loss_dict,on_step=True,on_epoch=True,prog_bar=True,prefix='train')

        self.log(
            "global_step", self.global_step, on_step=True, on_epoch=False, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss_dict['loss']


    def on_train_batch_end(
        self, outputs: th.Tensor, batch: Any, batch_idx: int
    ) -> None:
        pass


    def predict_seg_mask(self, image):
        #predict the segmentation mask for a given image window
        start_time = time.time()
        B,_,W,H,D = image.shape
        C = self.hparams.extra_kwargs.num_targets
        mask_shape = (B,C,W,H,D)

        patch_sizes = self.hparams.extra_kwargs.patch_sizes

        #get patch embeddings and pred patch labels from SwinT encoder
        patches_embeddings, patches_pred_labels = self.forward(image=image,pred_type='patchify')
        patch_emb_size = self.hparams.extra_kwargs.patch_emb_size
        C_=patch_emb_size**3

        pred_masks = []

        #filter tumor patch locations using patches_pred_labels and generate masks only for these patch locations
        for i, patch_size in enumerate(patch_sizes):

            #patch locations
            W_,H_,D_ = (W//patch_size,H//patch_size,D//patch_size)

            #get tumor patch indices
            patch_pred_labels = patches_pred_labels[patch_size].sigmoid().gt(self.hparams.extra_kwargs.patch_thresh)
            assert patch_pred_labels.shape == (B,1,W_,H_,D_)
            tumor_patch_indices = patch_pred_labels.nonzero(as_tuple=True)
            num_tumor_patches = tumor_patch_indices[0].shape[0]

            pred_mask = torch.zeros(B, C, W, H, D).to(image)
            if num_tumor_patches > 0:
                #get patch_embeddings for the current patch_size
                patch_embeddings = patches_embeddings[patch_size]
                assert patch_embeddings.shape == (B, C_, W_, H_, D_)

                #get the patch_embeddings corresponding to tumor patch indices
                #B,C_,W_,H_,D_ -> (num_tumor_patches ,C_)
                tumor_patch_embeddings = patch_embeddings[tumor_patch_indices[0], :, tumor_patch_indices[2], tumor_patch_indices[3], tumor_patch_indices[4]]

                #tumor_patch_embeddings: (num_tumor_patches, C_) -> (num_tumor_patches, 1, patch_emb_size, patch_emb_size, patch_emb_size)
                tumor_patch_embeddings = tumor_patch_embeddings.view(-1, 1, patch_emb_size, patch_emb_size, patch_emb_size)
                assert tumor_patch_embeddings.shape == (num_tumor_patches, 1, patch_emb_size, patch_emb_size, patch_emb_size)

                #upsample the patch embeddings from patch_emb_size to match the patch_size resolutions and embed the upsampled patches
                patch, embeddings = self.forward(patch_embeddings=tumor_patch_embeddings, patch_size=patch_size, pred_type="up_and_embed")
                assert patch.shape  == (num_tumor_patches, image.shape[1], patch_size, patch_size, patch_size)

                #sample the tumor mask patches using ddim with tumor_patch_embeddings as condition
                mask_patch_shape = (num_tumor_patches, C, patch_size, patch_size, patch_size)
                pred_tumor_mask_patch = self.forward(pred_type=self.hparams.extra_kwargs.sampling_type, mask_patch_shape=mask_patch_shape, \
                                                        patch=patch, embeddings=embeddings, patch_size=patch_size)

                #reassemble the predicted tumor mask patches to get the tumor mask for the entire window
                #----create a zero tensor for the window mask, and fill in the tumor patch locations with the corresponding predicted mask patches
                fill_indices = patch_pred_labels.nonzero(as_tuple=False)
                for j in range(num_tumor_patches):
                    b, _, w_idx, h_idx, d_idx = fill_indices[j]
                    pred_mask[b, :, w_idx * patch_size:(w_idx + 1) * patch_size,
                                    h_idx * patch_size:(h_idx + 1) * patch_size, 
                                    d_idx * patch_size:(d_idx + 1) * patch_size] = pred_tumor_mask_patch[j]

            pred_masks.append(pred_mask)

        out = {f'res={patch_sizes[i]}':pred_masks[i] for i in range(len(patch_sizes))}

        #Mean of mask window predictions using all the 
        mean_pred_mask = torch.mean(torch.stack(pred_masks), dim=0)
        out['res=mean'] = mean_pred_mask
        return out


    def validation_step(self, batch):
        start_time = time.time()

        image, mask, foreground, patch_tumor_labels = batch["image"], batch["mask"], batch["foreground"], batch["patch_tumor_labels"]
        B,C,W,H,D = mask.shape
        mask = mask.float()
        subregions_names = self.hparams.extra_kwargs.subregions_names

        val_metrics = {}
        #get patch embeddings and pred patch labels from SwinT encoder
        #compute patch classify loss
        patches_embeddings, patches_pred_labels = self.forward(image=image,pred_type='patchify')
        patches_classify_loss_dict = self.patch_criterion(patches_pred_labels, patch_tumor_labels)
        val_metrics.update(patches_classify_loss_dict)

        #compute patch classification metrics on binarized logits (dice score)
        patch_sizes = self.hparams.extra_kwargs.patch_sizes
        patch_classify_dice = 0.0
        for patch_size in patch_sizes:
            pred_scores, dice = compute_subregions_pred_metrics(patches_pred_labels[patch_size], patch_tumor_labels[patch_size], 1, \
                                                        subregions_names, prefix_key='patch_classify', suffix_key = {'res' : patch_size})
            val_metrics.update(pred_scores)
            patch_classify_dice += dice
        patch_classify_dice /= len(patch_sizes)
        val_metrics['patch_classify_dice'] = patch_classify_dice

        seg_metrics = {'seg_loss':0, 'seg_dice':0}
        pred_masks = []
        patch_emb_size = self.hparams.extra_kwargs.patch_emb_size
        C_=patch_emb_size**3

        #filter tumor patch locations using patches_pred_labels and generate masks only for these patch locations
        for i, patch_size in enumerate(patch_sizes):

            #patch locations
            W_,H_,D_ = (W//patch_size,H//patch_size,D//patch_size)

            #get tumor patch indices
            patch_pred_labels = patches_pred_labels[patch_size].sigmoid().gt(self.hparams.extra_kwargs.patch_thresh)
            assert patch_pred_labels.shape == (B,1,W_,H_,D_)
            tumor_patch_indices = patch_pred_labels.nonzero(as_tuple=True)
            num_tumor_patches = tumor_patch_indices[0].shape[0]

            pred_mask = torch.zeros(B,C,W,H,D).to(mask)

            if num_tumor_patches > 0:
                #get patch_embeddings for the current patch_size
                patch_embeddings = patches_embeddings[patch_size]
                assert patch_embeddings.shape == (B, C_, W_, H_, D_)

                #get the patch_embeddings corresponding to tumor patch indices
                #B,C_,W_,H_,D_ -> (num_tumor_patches ,C_)
                tumor_patch_embeddings = patch_embeddings[tumor_patch_indices[0], :, tumor_patch_indices[2], tumor_patch_indices[3], tumor_patch_indices[4]]

                #tumor_patch_embeddings: (num_tumor_patches, C_) -> (num_tumor_patches, 1, patch_emb_size, patch_emb_size, patch_emb_size)
                tumor_patch_embeddings = tumor_patch_embeddings.view(-1, 1, patch_emb_size, patch_emb_size, patch_emb_size)
                assert tumor_patch_embeddings.shape == (num_tumor_patches, 1, patch_emb_size, patch_emb_size, patch_emb_size)

                #upsample the patch embeddings from patch_emb_size to match the patch_size resolutions and embed the upsampled patches
                patch, embeddings = self.forward(patch_embeddings=tumor_patch_embeddings, patch_size=patch_size, pred_type="up_and_embed")
                assert patch.shape  == (num_tumor_patches, image.shape[1], patch_size, patch_size, patch_size)

                #sample the tumor mask patches using ddim with tumor_patch_embeddings as condition
                mask_patch_shape = (num_tumor_patches, C, patch_size, patch_size, patch_size)
                pred_tumor_mask_patch = self.forward(pred_type=self.hparams.extra_kwargs.sampling_type, mask_patch_shape=mask_patch_shape, \
                                                        patch=patch, embeddings=embeddings, patch_size=patch_size)

                #reassemble the predicted tumor mask patches to get the tumor mask for the entire window
                #----create a zero tensor for the window mask, and fill in the tumor patch locations with the corresponding predicted mask patches
                pred_mask = torch.zeros(B,C,W,H,D).to(mask)
                fill_indices = patch_pred_labels.nonzero(as_tuple=False)
                for j in range(num_tumor_patches):
                    b, _, w_idx, h_idx, d_idx = fill_indices[j]
                    pred_mask[b, :, w_idx * patch_size:(w_idx + 1) * patch_size,
                                    h_idx * patch_size:(h_idx + 1) * patch_size, 
                                    d_idx * patch_size:(d_idx + 1) * patch_size] = pred_tumor_mask_patch[j]

            pred_masks.append(pred_mask)

            #compute segmentation loss
            seg_losses = self.criterion(pred_mask * foreground, mask)
            seg_metrics[f"seg_dice_loss_res={patch_size}"] = seg_losses['dice_loss']
            seg_metrics[f"seg_ce_loss_res={patch_size}"] = seg_losses['bce_loss']
            seg_metrics[f"seg_loss_res={patch_size}"] = seg_losses['dice_loss'] + seg_losses['bce_loss']
            seg_metrics['seg_loss'] += seg_metrics[f"seg_loss_res={patch_size}"]

            #compute segmentation metrics
            pred_scores, dice = compute_subregions_pred_metrics(pred_mask*foreground,mask,C,subregions_names,prefix_key='seg',suffix_key={'res':patch_size})
            seg_metrics['seg_dice'] += dice
            seg_metrics.update(pred_scores)


        #Mean of mask window predictions using all the patch_sizes
        patch_size = 'mean'
        mean_pred_mask = torch.mean(torch.stack(pred_masks), dim=0)

        mean_pred_seg_losses = self.criterion(mean_pred_mask * foreground, mask)
        seg_metrics[f"seg_dice_loss_res={patch_size}"] = mean_pred_seg_losses['dice_loss']
        seg_metrics[f"seg_ce_loss_res={patch_size}"] = mean_pred_seg_losses['bce_loss']
        seg_metrics[f"seg_loss_res={patch_size}"] = mean_pred_seg_losses['dice_loss'] + mean_pred_seg_losses['bce_loss']
        seg_metrics['seg_loss'] += seg_metrics[f"seg_loss_res={patch_size}"]
        seg_metrics['seg_loss'] /= (len(patch_sizes)+1)

        #compute mean_pred segmentation metrics
        mean_pred_scores, mean_pred_dice = compute_subregions_pred_metrics(mean_pred_mask*foreground,mask,C,subregions_names,prefix_key='seg',suffix_key={'res':patch_size})
        seg_metrics.update(mean_pred_scores)
        seg_metrics['seg_dice'] += mean_pred_dice
        seg_metrics['seg_dice'] /= (len(patch_sizes)+1)

        val_metrics.update(seg_metrics)

        self._log_scores(val_metrics,prefix='val',on_epoch=True,prog_bar=True)
        self.log(f"val/dice", seg_metrics['seg_dice'], on_step=False, on_epoch=True, prog_bar=True)

        classify_and_seg_dice = (val_metrics['patch_classify_dice'] + val_metrics['seg_dice'])/2
        self.log(f"val/class+seg_dice", classify_and_seg_dice, on_step=False, on_epoch=True, prog_bar=True)

        dur =  (time.time() - start_time)/60
        log.info(f"Validation took {dur:0.4f} mins")




    def test_step(self, batch: Any, batch_idx: int):
        data, file_id = batch
        image, mask, foreground = data["image"], data["mask"], data["foreground"]
        B,_,W,H,D = image.shape

        pred_seg_masks = self.inferer(inputs=image, network=self.predict_seg_mask)
        #pred_seg_masks = {'res=16':torch.randn(B,3,W,H,D),'res=32':torch.randn(B,3,W,H,D),'res=mean':torch.randn(B,3,W,H,D)}

        subregions_names = self.hparams.extra_kwargs.subregions_names
        #patch_sizes = self.trainer.datamodule.hparams.patch_sizes
        C = self.hparams.extra_kwargs.num_targets

        test_metrics = {}
        test_dice = 0
        for key in pred_seg_masks.keys():
            pred_mask = pred_seg_masks[key]*foreground
            seg_losses = self.criterion(pred_mask, mask)
            loss_dice = seg_losses['dice_loss']
            loss_bce = seg_losses['bce_loss']
            loss = loss_dice + loss_bce
            test_metrics[f"dice_loss_{key}"] = loss_dice
            test_metrics[f"bce_loss_{key}"] = loss_bce
            test_metrics[f"loss_{key}"] = loss

            pred_scores, pred_dice = compute_subregions_pred_metrics(pred_mask,mask,C,subregions_names,suffix_key=key)
            test_metrics.update(pred_scores)
            test_dice += pred_dice
            pred_seg_masks[key] = pred_mask.sigmoid().gt(0.5)

        test_dice /= len(pred_seg_masks)
        self._log_scores(test_metrics,prefix='test',on_epoch=True,prog_bar=True)
        self.log(f"test/dice", test_dice, on_step=False, on_epoch=True, prog_bar=True)

        return pred_seg_masks, file_id



    def predict_step(self, batch):
        data, file_ids = batch
        image, foreground = data["image"], data["foreground"]
        pred_seg_masks = self.inferer(inputs=image, network=self.predict_seg_mask)
        for key in pred_seg_masks.keys():
            pred_mask = pred_seg_masks[key]*foreground
            pred_seg_masks[key] = pred_mask.sigmoid().gt(0.5)
        return pred_seg_masks,file_ids



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
            self.denoise_net = th.compile(self.denoise_net)


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
