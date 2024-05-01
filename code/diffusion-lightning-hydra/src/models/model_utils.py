from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.guided_diffusion import gaussian_diffusion as gd
from src.guided_diffusion.respace import SpacedDiffusion, space_timesteps
from src.guided_diffusion.unet import SuperResModel, UNetModel, EncoderUNetModel
from src.guided_diffusion.resample import ScheduleSampler

from dataclasses import dataclass, Field

from src.guided_diffusion.resample import create_named_schedule_sampler


def zero_grad(params):
    for param in params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

def params_to_state_dict(net,params):
    state_dict = net.state_dict()
    for i, (name, _value) in enumerate(net.named_parameters()):
        assert name in state_dict
        state_dict[name] = params[i]
    return state_dict


def state_dict_to_params(net,state_dict):
    params = [state_dict[name] for name, _ in net.named_parameters()]
    return params

    
def create_model(
    image_size:int=64,
    num_channels:int=128,
    num_res_blocks:int=2,
    channel_mult:str="",
    learn_sigma:bool=False,
    class_cond:bool=False,
    num_classes:int=None,
    use_checkpoint:bool=False,
    attention_resolutions:str="16",
    num_heads:int=1,
    num_head_channels:int=-1,
    num_heads_upsample:int=-1,
    use_scale_shift_norm:bool=False,
    dropout:int=0,
    resblock_updown:bool=False,
    use_fp16:bool=False,
    use_new_attention_order:bool=False
    
)-> UNetModel:

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
    steps: int=1000,
    learn_sigma: bool=False,
    sigma_small: bool=False,
    noise_schedule: str ="linear",
    use_kl: bool=False,
    predict_xstart: bool=False,
    rescale_timesteps: bool=False,
    rescale_learned_sigmas: bool=False,
    timestep_respacing: str=""

)-> SpacedDiffusion:


    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


