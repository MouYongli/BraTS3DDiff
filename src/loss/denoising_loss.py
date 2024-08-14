import torch.nn as nn
from src.models.diffusion.gaussian_diffusion import GaussianDiffusion
from src.models.diffusion.enums import *
from src.models.utils.utils import mean_flat
import torch

class DenoisingLoss(nn.Module):
    def __init__(self,diffusion:GaussianDiffusion):
        super(DenoisingLoss, self).__init__()
        self.diffusion = diffusion

    def _simple_mse_loss(self,model_output,x_start,noise):
        if self.diffusion.model_mean_type == ModelMeanType.START_X:
            target = x_start
        elif self.diffusion.model_mean_type == ModelMeanType.EPSILON:
            target = noise
        else:
            raise NotImplementedError(self.diffusion.model_mean_type)

        assert model_output.shape == target.shape == x_start.shape
        mse = mean_flat((target - model_output) ** 2)
        return mse


    def _vlb_loss(self,model_output,x_start,x_t,t):
        return self.diffusion._vb_terms_bpd(
                model=lambda *args, r=model_output: r,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs={},
            )["output"]


    def _denoising_loss(self,model_output,x_start,x_t,t,noise):
        loss_terms={}

        loss_terms["mse"] = self._simple_mse_loss(model_output,x_start,noise)

        if self.diffusion.model_var_type in [ModelVarType.LEARNED_RANGE, ModelVarType.LEARNED]:
            B, C = x_t.shape[:2]
            assert model_output.shape == (B, C * 2, *x_t.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
            loss_terms["vb"] = self._vlb_loss(frozen_out,x_start,x_t,t)
            if self.diffusion.loss_type == LossType.RESCALED_MSE:
                # Divide by 1000 for equivalence with initial implementation.
                # Without a factor of 1/1000, the VB term hurts the MSE term.
                loss_terms["vb"] *= self.diffusion.num_timesteps / 1000.0

        if "vb" in loss_terms:
            loss_terms["loss"] = loss_terms["mse"] + loss_terms["vb"]
        else:
            loss_terms["loss"] = loss_terms["mse"]
        return loss_terms


    def forward(self,model_output,x_start,x_t,t,noise):
        assert self.diffusion.loss_type in [LossType.MSE,LossType.RESCALED_MSE]
        return self._denoising_loss(model_output,x_start,x_t,t,noise)