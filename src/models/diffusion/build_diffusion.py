from typing import Any, Callable

import torch

from .enums import *
from .noise_schedule import get_named_beta_schedule
from .respace import SpacedDiffusion, space_timesteps


class BuildDiffusion(torch.nn.Module):
    def __init__(
        self,
        steps: int = 1000,
        sample_steps: int = 50,
        noise_schedule: str = "linear",
        model_mean_type: str = "START_X",
        model_var_type: str = "FIXED_LARGE",
        loss_type: str = "MSE",
    ):
        super().__init__()

        betas = get_named_beta_schedule(noise_schedule, steps)
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(steps, [steps]),
            betas=betas,
            model_mean_type=ModelMeanType[model_mean_type],
            model_var_type=ModelVarType[model_var_type],
            loss_type=LossType[loss_type],
        )

        self.sample_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(steps, [sample_steps]),
            betas=betas,
            model_mean_type=ModelMeanType[model_mean_type],
            model_var_type=ModelVarType[model_var_type],
            loss_type=LossType[loss_type],
        )
        self._validate_args()

    def _validate_args(self):
        assert self.diffusion.loss_type in [LossType.MSE, LossType.RESCALED_MSE]
        assert self.diffusion.model_mean_type in [
            ModelMeanType.START_X,
            ModelMeanType.EPSILON,
        ]
        assert self.diffusion.model_var_type in [
            ModelVarType.FIXED_LARGE,
            ModelVarType.FIXED_SMALL,
            ModelVarType.LEARNED_RANGE,
        ]
