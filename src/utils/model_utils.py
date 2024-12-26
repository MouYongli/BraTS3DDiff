from operator import itemgetter

import torch
import torch.distributed as dist
from typing import Sequence
import numpy as np

from monai.metrics import DiceMetric, confusion_matrix

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
        uncer_out = 0
        for i in range(uncer_step):
            uncer_out += sample_outputs[i]["all_model_outputs"][index]
        uncer_out = uncer_out / uncer_step
        uncer = compute_uncer(uncer_out).cpu()
        w = torch.exp(
            torch.sigmoid(torch.tensor((index + 1) / num_sample_timesteps))
            * (1 - uncer)
        )
        for i in range(uncer_step):
            sample_return += w * sample_outputs[i]["all_samples"][index].cpu()

    return sample_return





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
        patch_map.permute(0, 2, 3, 4, 1)
        .contiguous()
        .view(-1, patch_channels, patch_size, patch_size, patch_size)
    )
    n_patches = B * W_ * H_ * D_
    assert patches.shape[0] == n_patches
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


if __name__ == "__main__":
    B,C_,W_,H_,D_ = (1,512,4,4,4)
    patch_labels = torch.rand(B,1,W_,H_,D_) > 1
    patch_map = torch.randn(B,C_,W_,H_,D_)
    get_nonzero_patches(patch_labels, patch_map, patch_size=8, patch_channels=1)


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







