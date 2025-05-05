from operator import itemgetter

import torch
import torch.functional as F

import torch.distributed as dist
from typing import Sequence
import numpy as np
import itertools

from monai.metrics import DiceMetric

def perfect_cbrt(n : Sequence[int]):
    #find cube_root of a seq of perfect cubes 
    # and throw error if any of the inputs is not a perfect cube
    cube_root = np.round(np.cbrt(n)).astype(int)
    is_cube = np.array_equal(cube_root ** 3, n)
    if not is_cube:
        raise ValueError(f"Input sequence {n} is not a perfect cube")
    else:
        return cube_root


def zero_grad(params):
    for param in params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def params_to_state_dict(net, params):
    state_dict = net.state_dict()
    for i, (name, _value) in enumerate(net.named_parameters()):
        assert name in state_dict
        state_dict[name] = params[i]
    return state_dict


def state_dict_to_params(net, state_dict):
    params = [state_dict[name] for name, _ in net.named_parameters()]
    return params


def update_ema(target_params, source_params, rate=0.99):
    """Update target parameters to be closer to those of source parameters using an exponential
    moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """Scale the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def aggregate_timestep_quantile_losses(qt_losses_dict):
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        qt_losses_dict_all = [None for _ in range(world_size)]
        dist.all_gather_object(qt_losses_dict_all, qt_losses_dict)

        qt_losses_dict_0 = qt_losses_dict_all.pop(0)
        for key in qt_losses_dict_0.keys():
            for i in range(len(qt_losses_dict_all)):
                # if key is quartile loss
                if key.split("_")[-1].startswith("q"):
                    count_0, avg_0 = itemgetter("count", "avg")(qt_losses_dict_0[key])
                    if key in qt_losses_dict_all[i]:
                        count_i, avg_i = itemgetter("count", "avg")(
                            qt_losses_dict_all[i][key]
                        )
                    else:
                        count_i, avg_i = (0, 0)
                    qt_losses_dict_0[key]["count"] += count_i
                    qt_losses_dict_0[key]["avg"] = (
                        (count_0 * avg_0) + (count_i * avg_i)
                    ) / qt_losses_dict_0[key]["count"]

                # if key is overall loss
                else:
                    qt_losses_dict_0[key] = 0.5 * (
                        qt_losses_dict_all[i][key] + qt_losses_dict_0[key]
                    )
    else:
        qt_losses_dict_0 = qt_losses_dict

    for key in qt_losses_dict_0.keys():
        if key.split("_")[-1].startswith("q"):
            qt_losses_dict_0[key] = qt_losses_dict_0[key]["avg"]

    return qt_losses_dict_0


def get_timestep_quantile_losses(ts, weights, losses, num_timesteps, qt_losses_dict):
    # computes an avg of losses, plus an running avg of losses for each timestep quantile (0-3)
    # adds the avgs to the previous qt_losses_dict
    if not qt_losses_dict:
        qt_losses_dict = {}

    for loss_term, loss_ts in losses.items():
        loss_ts = loss_ts * weights
        loss = loss_ts.mean().item()
        if loss_term not in qt_losses_dict:
            qt_losses_dict[loss_term] = loss
        else:
            qt_losses_dict[loss_term] = 0.5 * (qt_losses_dict[loss_term] + loss)

        for t, loss_t in zip(ts, loss_ts):
            t_quartile = int(4 * t / num_timesteps)
            key = f"{loss_term}_q{t_quartile}"
            if key not in qt_losses_dict:
                qt_losses_dict[key] = {"avg": loss_t.item(), "count": 1}
            else:
                count, avg = itemgetter("count", "avg")(qt_losses_dict[key])
                qt_losses_dict[key]["count"] += 1
                qt_losses_dict[key]["avg"] = (
                    (avg * count) + loss_t.item()
                ) / qt_losses_dict[key]["count"]

    return qt_losses_dict


def compute_segmentation_metrics(
    y_logits, y_true, C, subregions_names, prefix_key=None, suffix_key=None, thresh=0.50
):
    # expects non-binarized y_logits
    # C = #subregions

    y_pred = y_logits.sigmoid().gt(thresh)
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean_batch",
        get_not_nans=True,
        ignore_empty=False,
    )
    # scores = {'dice':0.0,'hd95':0.0,'recall':0.0}

    if prefix_key is not None:
        if type(prefix_key) == dict:
            # suffix_key = "key1=val1-key2=val2-"
            prefix_key = "-".join([f"{k}={v}" for k, v in prefix_key.items()])
        prefix_key = f"{prefix_key}-"
    else:
        prefix_key = ""

    if suffix_key is not None:
        if type(suffix_key) == dict:
            # suffix_key = "-key1=val1-key2=val2"
            suffix_key = "-".join([f"{k}={v}" for k, v in suffix_key.items()])
        suffix_key = f"-{suffix_key}"
    else:
        suffix_key = ""

    scores = {f"{prefix_key}dice{suffix_key}": 0.0}

    for c in range(C):
        scores[f"{prefix_key}dice_{subregions_names[c]}{suffix_key}"] = dice_metric(
            y_pred[:, c].unsqueeze(1), y_true[:, c].unsqueeze(1)
        ).mean()
        # scores[f"hd95_{subregions_names[c]}"] = hausdorff_distance_95(y_pred[:, c].unsqueeze(1), y_true[:, c].unsqueeze(1))
        # scores[f"recall_{subregions_names[c]}"] = recall(y_pred[:, c].unsqueeze(1), y_true[:, c].unsqueeze(1))

        scores[f"{prefix_key}dice{suffix_key}"] += scores[
            f"{prefix_key}dice_{subregions_names[c]}{suffix_key}"
        ]
        # scores[f"hd95"] += scores[f"hd95_{subregions_names[c]}"]
        # scores[f"recall"] += scores[f"recall_{subregions_names[c]}"]

    scores[f"{prefix_key}dice{suffix_key}"] /= C
    # scores[f"hd95"] /= C
    # scores[f"recall"] /= C
    return scores, scores[f"{prefix_key}dice{suffix_key}"]



def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = -pred_out * torch.log(pred_out)
    return uncer_out


def compute_uncertainty_based_fusion(
    sample_outputs, out_shape, uncer_step=4, num_sample_timesteps=10
):
    # Adapted from DIffUNet
    sample_return = torch.zeros(out_shape)
    for index in range(num_sample_timesteps):
        #compute uncertainity in each timestep based on model output
        uncer_out = 0
        for i in range(uncer_step):
            uncer_out += sample_outputs[i]["model_outputs"][index]
        uncer_out = uncer_out / uncer_step
        uncer = compute_uncer(uncer_out).cpu()
        w = torch.exp(
            torch.sigmoid(torch.tensor((index + 1) / num_sample_timesteps))
            * (1 - uncer)
        )
        #final sample: 
        #   weighted average of pred_xstarts from every timestep based on uncertainity in every timestep
        for i in range(uncer_step):
            sample_return += w * sample_outputs[i]["pred_xstarts"][index].cpu()

    return sample_return


def add_background_test(orig_img:torch.Tensor, fg_crop:torch.Tensor, fg_start:Sequence[int], fg_end:Sequence[int]):
    img_with_bg = torch.zeros_like(orig_img)
    img_with_bg[:,fg_start[0]: fg_end[0], fg_start[1]: fg_end[1], fg_start[2]: fg_end[2]] = fg_crop
    assert torch.all(img_with_bg==orig_img)
    return img_with_bg

def add_background(fg:torch.Tensor, orig_img_shape:Sequence[int], fg_start:Sequence[int], fg_end:Sequence[int]):
    img_with_bg = torch.zeros(orig_img_shape).to(fg)
    img_with_bg[:,fg_start[0]: fg_end[0], fg_start[1]: fg_end[1], fg_start[2]: fg_end[2]] = fg
    return img_with_bg

def add_background_batch(fg:torch.Tensor, orig_img_shape:Sequence[int], fg_start:Sequence[int], fg_end:Sequence[int]):
    assert fg.shape[0] == 1
    return add_background(fg.squeeze(0), orig_img_shape[0], fg_start[0], fg_end[0]).unsqueeze(0)


def labels2onehot(mask_labels, num_classes=4):
    if len(mask_labels.shape) == 3:
        c, w, h = mask_labels.shape
    elif len(mask_labels.shape) == 4:
        c, w, h, d = mask_labels.shape
    elif len(mask_labels.shape) == 5:
        b, c, w, h, d = mask_labels.shape
    else:
        raise ValueError
    assert c == 1
    if len(mask_labels.shape) == 3:
        mask_onehot_labels =  F.one_hot(mask_labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).contiguous().view(-1, w, h)[1:, :, :]
    if len(mask_labels.shape) == 4:
        mask_onehot_labels =  F.one_hot(mask_labels.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).contiguous().view(-1, w, h, d)[1:, :, :, :]
    elif len(mask_labels.shape) == 5:
        mask_onehot_labels = F.one_hot(mask_labels.long(), num_classes=num_classes).permute(0, 1, 5, 2, 3, 4).contiguous().view(b, -1, w, h, d)[:, 1:, :, :, :]
    assert len(mask_labels.shape) == len(mask_onehot_labels.shape)
    mask_onehot_labels = mask_onehot_labels.to(torch.uint8)
    return mask_onehot_labels


def window2patches(win, patch_size=16):
    '''
    Divide a (B,C,W,H,D) sized window into (C,P,P,P) sized patches,
    and add all the patches in the batch dimension

    Args:
        win: window tensor (shape: (B, C, W, H, D))
        patch_size: patch resolution P, scalar
    Returns:
        patches: a batch of patches (N, C, P, P, P),
                where N=num of patches=(B*W//P*H//P*D//P)
    '''
    B, C, W, H, D = win.shape
    W_, H_, D_ = (W // patch_size, H // patch_size, D // patch_size)
    patches = win.view(B, C, W_, patch_size, H_, patch_size, D_, patch_size)
    patches = (
        patches.permute(0, 2, 4, 6, 1, 3, 5, 7)
        .contiguous()
        .view(-1, C, patch_size, patch_size, patch_size)
    )
    n_patches = B * W_ * H_ * D_
    assert patches.shape[0] == n_patches
    return patches



def patches2window(patches, win_size=(128,128,128)):
    '''
    Assemble all patches to get the whole window
    # (B*W_*H_,D_,C,patch_size,patch_size,patch_size) ->
    # (B,C,W_,patch_size,H_,patch_size,D_,patch_size) ->
    # (B,C,W,H,D)

    Args:
        patches: a batch of patches (N, C, P, P, P),
                where N=num of patches=(B*W//P*H//P*D//P)
        patch_size: patch resolution P, scalar
    Returns:
        win: window tensor (shape: (B, C, W, H, D))

    '''
    _, C, patch_size, patch_size, patch_size = patches.shape
    W, H, D = win_size
    W_, H_, D_ = (W // patch_size, H // patch_size, D // patch_size)
    win = (
        patches.view(
            -1, W_, H_, D_, C, patch_size, patch_size, patch_size
        )
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .contiguous()
        .view(-1, C, W, H, D)
    )
    return win


def stable_divide(nom, denom):
    return nom/denom if denom != 0 else 0


def expand_patches(patch_labels, patch_size=16, patch_channels=3):
    '''
    Expand 3D tensor of patch_labels (B,1,W_,H_,D_) such that,
    every patch label (scalar value) gets expanded to the size of the patch = (C,P,P,P)
    
    Output patch_labels shape: B,C,W,H,D
        (B,1,W_,H_,D_) -> (B,1,W_,1,H_,1,D_,1) -> B,C,W_,P,H_,P,D_,P -> B,C,W,H,D
    '''
    # Expand shape of patch_labels so that it matches the shape of pred_mask
    # patch_pred_labels (B,1,W_,H_,D_) -> (B,1,W_,1,H_,1,D_,1) -> B,C,W_,P,H_,P,D_,P -> B,C,W,H,D
    B, C_, W_, H_, D_ = patch_labels.shape
    assert C_==1
    patch_labels = (
        patch_labels.view(B, 1, W_, 1, H_, 1, D_, 1)
        .expand(B, patch_channels, W_, patch_size, H_, patch_size, D_, patch_size)
        .contiguous()
        .view(B, patch_channels, W_*patch_size, H_*patch_size, D_*patch_size)
    )
    return patch_labels


def get_all_patches_1D(patch_map):
    '''
    Reshape a tensor of patch features so that all patches get added in the batch dimension 
    Output: Batch of 3D patches
    Args:
        patch_map: 3D tensor of patch features (shape: B,C_,W_,H_,D_))
                (W_,H_,D_) indexes the patch location and C_ corresponds to the 1D feature of every patch
    Returns:
        patches: A batch of 3D patches (N, C_),
                where N=num of patches=(B*W_*H_*D_)
    '''

    B, C_, W_, H_, D_ = patch_map.shape
    patches = (
        patch_map.permute(0, 2, 3, 4, 1)
        .contiguous()
        .view(-1, C_)
    )
    n_patches = B * W_ * H_ * D_
    assert patches.shape[0] == n_patches
    return patches


def get_all_patches(patch_map, patch_size=8, patch_channels=1):
    '''
    Reshape a tensor of patch features so that all patches get added in the batch dimension 
    Output: Batch of 3D patches
    Args:
        patch_map: 3D tensor of patch features (shape: B,C_,W_,H_,D_))
                (W_,H_,D_) indexes the patch location and C_ corresponds to the 1D feature of every patch
        patch_size: patch resolution P, scalar
        patch_channels: number of patch channels C
            1D patch features should able to be reshaped to a 3D tensor, ie C_==C*(P**3) 
    Returns:
        patches: A batch of 3D patches (N, C, P, P, P),
                where N=num of patches=(B*W_*H_*D_)
    '''

    B, C_, W_, H_, D_ = patch_map.shape
    assert C_ == patch_channels*(patch_size**3)
    patches = (
        get_all_patches_1D(patch_map)
        .view(-1, patch_channels, patch_size, patch_size, patch_size)
    )
    return patches

def get_zero_patches(patch_labels, patch_map, patch_size=8, patch_channels=1):
    '''Get the patches where patch_labels == 0
    '''
    return get_nonzero_patches(~patch_labels, patch_map, patch_size, patch_channels)


def get_nonzero_patches(patch_labels, patch_map, patch_size=8, patch_channels=1):
    '''
    Get the patches where patch_labels == 1
    Args:
        patch_labels: bool tensor (shape: (B, 1, W_, H_, D_))
        patch_map: float tensor (shape: (B, C*P^3, W_, H_, D_))
                3D map of 1D patch tensors. Every spatial location indicates a patch location,
                channels are patch features
    Returns:
        nonzero_patches: float tensor (shape: (num_nonzero_patches, C, P, P, P))
    '''
    B, C_, W_, H_, D_ = patch_map.shape
    assert C_ == patch_channels*(patch_size**3)

    # Find nonzero patches
    nonzero_patch_indices = patch_labels.nonzero(as_tuple=True)
    num_nonzero_patches = nonzero_patch_indices[0].shape[0]
    if num_nonzero_patches == 0:
        return (0, None)

    # get the patch data corresponding to nonzero patch indices
    # B,C*P*P*P,W_,H_,D_ -> (num_nonzero_patches ,C*P*P*P)
    nonzero_patches = patch_map[
        
        nonzero_patch_indices[0],
        :,
        nonzero_patch_indices[2],
        nonzero_patch_indices[3],
        nonzero_patch_indices[4],
    ]

    # nonzero_patch: (num_nonzero_patches, C*P*P*P) -> (num_nonzero_patches, C, patch_size, patch_size, patch_size)
    nonzero_patches = nonzero_patches.view(
        num_nonzero_patches, patch_channels, patch_size, patch_size, patch_size
    )
    
    return (num_nonzero_patches, nonzero_patches)


def get_patches_from_idx(patch_map, patch_idxs, patch_size=8, patch_channels=1):
    '''
    Get the patches where patch_labels == 1
    Args:
        patch_labels: bool tensor (shape: (B, 1, W_, H_, D_))
        patch_map: float tensor (shape: (B, C*P^3, W_, H_, D_))
                3D map of 1D patch tensors. Every spatial location indicates a patch location,
                channels are patch features
    Returns:
        nonzero_patches: float tensor (shape: (num_nonzero_patches, C, P, P, P))
    '''
    B, C_, W_, H_, D_ = patch_map.shape
    assert C_ == patch_channels*(patch_size**3)

    # Find nonzero patches
    nonzero_patch_indices = patch_labels.nonzero(as_tuple=True)
    num_nonzero_patches = nonzero_patch_indices[0].shape[0]
    if num_nonzero_patches == 0:
        return (0, None)

    # get the patch data corresponding to nonzero patch indices
    # B,C*P*P*P,W_,H_,D_ -> (num_nonzero_patches ,C*P*P*P)
    nonzero_patches = patch_map[
        nonzero_patch_indices[0],
        :,
        nonzero_patch_indices[2],
        nonzero_patch_indices[3],
        nonzero_patch_indices[4],
    ]

    # nonzero_patch: (num_nonzero_patches, C*P*P*P) -> (num_nonzero_patches, C, patch_size, patch_size, patch_size)
    nonzero_patches = nonzero_patches.view(
        num_nonzero_patches, patch_channels, patch_size, patch_size, patch_size
    )
    
    return (num_nonzero_patches, nonzero_patches)

def fill_in_window_with_patches(window,patch_labels,fill_patches):
    '''
    fill in a (zero tensor) window, at patch locations where patch_labels == 1
    with corresponding fill_patches data

    Args:
        window: tensor for the entire window (shape: (B, C, W, H, D))
            window tensor should be uninitialized ie zero at locations where patch_labels==1
        patch_labels: bool tensor (shape: (B, 1, W_, H_, D_)) indexes the patch locations, (W_= W//P, H_= H//P, D_= D//P)
        fill_patches: float tensor (shape: (X, C, P, P, P)) , patch data to fill in with
                where X is the number of patches where patch_labels==1
    Returns:
        nonzero_patches: float tensor (shape: (num_nonzero_patches, C, P, P, P))
    '''
    # fill in the zero tensor pred_mask created before, at the nonzero patch locations
    #  with the corresponding predicted mask patches

    patch_size = fill_patches.shape[2]
    assert window.shape[2] == patch_labels.shape[2]*patch_size  #W==W_*P

    fill_indices = patch_labels.nonzero(as_tuple=False)
    assert fill_patches.shape[0] == len(fill_indices)

    for j in range(len(fill_indices)):
        b, _, w_idx, h_idx, d_idx = fill_indices[j]
        #should not overwrite data, fill locations should be 0
        assert torch.all(window[
            b,
            :,
            w_idx * patch_size : (w_idx + 1) * patch_size,
            h_idx * patch_size : (h_idx + 1) * patch_size,
            d_idx * patch_size : (d_idx + 1) * patch_size,
        ] == 0).item()

        window[
            b,
            :,
            w_idx * patch_size : (w_idx + 1) * patch_size,
            h_idx * patch_size : (h_idx + 1) * patch_size,
            d_idx * patch_size : (d_idx + 1) * patch_size,
        ] = fill_patches[j]
    return window




def sample_patch_indices(patch_tumor_vol_fracs, eps=1e-20, num_samples=None):
    #patch_tumor_vol_fracs = (B,1,W_,H_,D_)
    wts = patch_tumor_vol_fracs.view(1,-1).squeeze(dim=0)
    wts = wts + eps
    if not num_samples:
        num_samples = wts.shape[0] // 2
    sampled_indices_flat = torch.multinomial(wts, num_samples=num_samples)
    sampled_indices_nd = torch.unravel_index(sampled_indices_flat, patch_tumor_vol_fracs.shape)
    return sampled_indices_flat, sampled_indices_nd    


def get_vals_from_idxs(x:torch.Tensor, idxs:tuple[torch.Tensor]):
    '''
    Args:
        x: (b,c,w,h,d) tensor. get values of given indexes along the c dimension
        idxs: idx of samples along each dimension
            ([....b_indices....],[.....c_indices....],[.....w_indices....],[.....h_indices....],[.....d_indices....])
    Returns: 
        Values of the selected samples along c dimension
        (num samples, c)
    '''
    return x[idxs[0], :, idxs[2], idxs[3], idxs[4]] #(num samples, C)


def ravel_index(index:torch.Tensor, shape:tuple[int]):
    """Ravel multi-dimensional indices to 1D index
    similar to np.ravel_multi_index
    Args:
        index (torch.tensor): indices in reversed order dn, ..., d1, with shape (..., n)
        shape (tuple): dn, ..., d1
    """
    #index = torch.tensor(index, dtype=torch.int64)
    shape = torch.tensor((1,) + shape[::-1], dtype=torch.int64).to(index) # =(1, d1, d1*d2, ..., d1*...*dn)
    shape = torch.cumprod(shape, dim=0)[:-1].flip(0) # =(d1*...*dn-1, ..., d1*d2, d1, 1)
    index = (index * shape).sum(dim=-1) # (...,)
    return index


def ravel_tuple_index(index:tuple[torch.Tensor], shape:tuple[int]):
    """Ravel multi-dimensional indices to 1D index
    similar to np.ravel_multi_index
    Args:
        index tuple(torch.tensor): each tuple item indices a certain dimension with shape (1, num_idxs)
        shape (tuple): dn, ..., d1
    """
    index = torch.stack(index,dim=0).T
    return ravel_index(index, shape)


def fold_patches(patches, img_shape, up:int=2):
    '''
    increase patch res by up x times
    img_shape: B, C, W, H, D
    Convert a tensor of all patches in img (B*W_*H_*D_, C, P, P, P) to 2x patches (B*W_*H_*D_//(up**3), C, up*P, up*P, up*P)
    W_, H_, D_ are the patch locations
    '''
    B, C, W, H, D = img_shape
    N, C, patch_size, patch_size, patch_size = patches.shape
    W_, H_, D_ = W//patch_size, H//patch_size, D//patch_size
    assert N == B*W_*H_*D_
    
    #target patch size and patch locations
    tgt_patch_size = up * patch_size
    tgt_W_, tgt_H_, tgt_D_  = W_//up, H_//up, D_//up

    return patches.view(B, W_, H_, D_, C, patch_size, patch_size, patch_size) \
        .permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous() \
        .view(B, C, tgt_W_, tgt_patch_size, tgt_H_, tgt_patch_size, tgt_D_, tgt_patch_size) \
        .permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous() \
        .view(-1, C, tgt_patch_size, tgt_patch_size, tgt_patch_size)



def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if (p.requires_grad) and (p.grad is not None):
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
    return total_norm


def plot_grad_flow(model):
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    norm_grads = []
    layers = []
    for n, p in model.named_parameters():
        if(p.requires_grad) and (p.grad is not None) and ("bias" not in n):
            layers.append(n)
            grad_ = p.grad.detach().cpu()
            ave_grads.append(grad_.abs().mean())
            max_grads.append(grad_.max())
            norm_grads.append(grad_.data.norm(2)/(grad_.data.numel()**(1/2)))
    plt.figure()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="red")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="green")
    #plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim() # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Gradient Stats")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="red", lw=4),
                Line2D([0], [0], color="green", lw=4)], ['max-gradient', 'mean-gradient'], labelcolor='black')
    #plt.tight_layout()
    fig = plt.gcf()
    fig.set_facecolor('white')
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.tight_layout()
    #fig.savefig(f'/home/ca550013/Master-Thesis/Projects/BraTS3DDiff/tmp/{tag}.png')
    plt.close('all')
    plt.clf()

    plt.figure()
    plt.bar(np.arange(len(max_grads)), norm_grads, alpha=0.2, lw=1, color="blue")
    #plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(max_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim() # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Normalized Gradient Norms")
    plt.title("Normalized Gradient Norms")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="blue", lw=4)], ['norm-gradient'], labelcolor='black')
    #plt.tight_layout()
    fig1 = plt.gcf()
    fig1.set_facecolor('white')
    fig1.set_size_inches(18.5, 10.5, forward=True)
    fig1.tight_layout()
    #fig1.savefig(f'/home/ca550013/Master-Thesis/Projects/BraTS3DDiff/tmp/{tag}_norms.png')
    plt.close('all')
    plt.clf()

    return fig, fig1

def get_x_patch_idx_from_2x_patch_idx_v1(patch_2x_flat_idx, patch_x_shape, patch_2x_shape):
  assert len(patch_x_shape) == len(patch_2x_shape) in [4, 5]
  ndim = len(patch_x_shape)
  assert np.all(np.array(patch_x_shape[2:]) == 2*np.array(patch_2x_shape[2:]))

  unravled_2x_idxs = torch.stack(torch.unravel_index(patch_2x_flat_idx, patch_2x_shape), dim=0).T
  x_start_offset = torch.tensor([1,1,2,2]) if ndim == 4 else torch.tensor([1,1,2,2,2])
  unravled_start_x_idxs = (unravled_2x_idxs*x_start_offset).to(unravled_2x_idxs)

  offsets = list(itertools.product([0, 1], repeat=ndim-2))
  offsets = [[0,0]+list(offset) for offset in offsets]
  offsets = torch.tensor(offsets).to(unravled_start_x_idxs)

  unravled_all_x_idxs = unravled_start_x_idxs.unsqueeze(1) + offsets
  unravled_all_x_idxs = unravled_all_x_idxs.contiguous().view(-1,len(patch_x_shape))

  flat_x_patch_idx = ravel_index(unravled_all_x_idxs, patch_x_shape)

  return flat_x_patch_idx

def get_x_patch_idx_from_2x_patch_idx_v2(patch_2x_flat_idx, patch_x_shape, patch_2x_shape):  
  assert len(patch_x_shape) == len(patch_2x_shape) in [4, 5]
  ndim = len(patch_x_shape)
  assert np.all(np.array(patch_x_shape[2:]) == 2*np.array(patch_2x_shape[2:]))
  
  num_patches_x = np.prod(patch_x_shape[2:])
  patch_x_idx_arr = torch.arange(num_patches_x).to(patch_2x_flat_idx)
  if ndim == 4:
    patch_x_idx_arr = patch_x_idx_arr.view(1,1,patch_x_shape[2],patch_x_shape[3])
    #patchify index array to 2x2 patches. a 2x2 patch stores the patch x indices of a 2x patch index
    patch_x_idx_arr = patch_x_idx_arr.view(1, 1, patch_x_shape[2]//2, 2, patch_x_shape[3]//2, 2) \
                                .permute(0, 2, 4, 1, 3, 5).contiguous().view(-1,1,2,2)
  else:
    patch_x_idx_arr = patch_x_idx_arr.view(1,1,patch_x_shape[2],patch_x_shape[3],patch_x_shape[4])
    #reshape index array to 2x2x2. a 2x2x2 patch stores the patch x indices of a 2x patch index
    patch_x_idx_arr = patch_x_idx_arr.view(1, 1, patch_x_shape[2]//2, 2, patch_x_shape[3]//2, 2, patch_x_shape[4]//2, 2) \
                                .permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(-1,1,2,2,2)
  
  patch_x_flat_idx = patch_x_idx_arr[patch_2x_flat_idx].flatten()
  return patch_x_flat_idx


def aggregate_common_x_and_2x_patches(img_patches_x, flat_patch_indices_x, patch_x_shape, img_patches_2x, flat_patch_indices_2x, patch_2x_shape, agg='mean'):
    '''
    Given a set of size x patches and size 2x patches corresponding to the same image,
    aggregate the common regions
    '''
    assert len(patch_x_shape) == len(patch_2x_shape) in [4, 5]
    assert np.all(np.array(patch_x_shape[2:]) == 2*np.array(patch_2x_shape[2:]))
    ndim = len(patch_x_shape)

    flat_patch_indices_x_from_2x = get_x_patch_idx_from_2x_patch_idx_v2(flat_patch_indices_2x, patch_x_shape, patch_2x_shape)

    if ndim == 4:
        N_x,C,x,x = img_patches_x.shape #N_x:num of size x patches
        assert img_patches_2x.shape[1:] == (C,2*x,2*x)
        img_patches_x_from_2x = img_patches_2x.view(-1,C,2,x,2,x).permute(0,2,4,1,3,5).contiguous().view(-1,C,x,x)
    else:
        N_x,C,x,x,x = img_patches_x.shape #N_x:num of size x patches
        assert img_patches_2x.shape[1:] == (C,2*x,2*x,2*x)
        img_patches_x_from_2x = img_patches_2x.view(-1,C,2,x,2,x,2,x).permute(0,2,4,6,1,3,5,7).contiguous().view(-1,C,x,x,x)

    common_patch_indices = (flat_patch_indices_x.unsqueeze(1)==flat_patch_indices_x_from_2x).nonzero()
    common_patches = torch.stack((img_patches_x[common_patch_indices[:,0]],  img_patches_x_from_2x[common_patch_indices[:,1]]))
    if agg=='mean':
        return common_patches.mean(dim=0)
    elif agg == 'sum':
        return common_patches.sum(dim=0)
    else:
        raise ValueError('agg should be either "mean" or "sum"')


def aggregate_common_x_and_2x_patches_mul(imgs_patches_x, flat_patch_indices_x, patch_x_shape, imgs_patches_2x, flat_patch_indices_2x, patch_2x_shape, agg='mean'):
    '''
    Given a set of size x patches and size 2x patches corresponding to the same image,
    aggregate the common regions
    '''
    assert np.all(np.array(patch_x_shape[2:]) == 2*np.array(patch_2x_shape[2:]))

    flat_patch_indices_x_from_2x = get_x_patch_idx_from_2x_patch_idx_v2(flat_patch_indices_2x, patch_x_shape, patch_2x_shape)
    common_patch_indices = (flat_patch_indices_x.unsqueeze(1)==flat_patch_indices_x_from_2x).nonzero()

    common_patches_list = []
    for img_patches_x, img_patches_2x in zip(imgs_patches_x, imgs_patches_2x):
        N_x,C,x,x,x = img_patches_x.shape #N_x:num of size x patches
        assert img_patches_2x.shape[1:] == (C,2*x,2*x,2*x)

        img_patches_x_from_2x = img_patches_2x.view(-1,C,2,x,2,x,2,x).permute(0,2,4,6,1,3,5,7).contiguous().view(-1,C,x,x,x)
        common_patches = torch.stack((img_patches_x[common_patch_indices[:,0]],  img_patches_x_from_2x[common_patch_indices[:,1]]))
        if agg=='mean':
            common_patches_list.append(common_patches.mean(dim=0))
        elif agg == 'sum':
            return common_patches_list.append(common_patches.sum(dim=0))
        else:
            raise ValueError('agg should be either "mean" or "sum"')

    return common_patches_list, common_patch_indices


if __name__ == "__main__":
    B,C_,W_,H_,D_ = (1,512,4,4,4)
    patch_labels = torch.rand(B,1,W_,H_,D_) > 1
    patch_map = torch.randn(B,C_,W_,H_,D_)
    get_nonzero_patches(patch_labels, patch_map, patch_size=8, patch_channels=1)










