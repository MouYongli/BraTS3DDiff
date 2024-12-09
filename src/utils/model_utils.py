from operator import itemgetter

import torch
import torch.distributed as dist
from monai.metrics import DiceMetric


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


def compute_subregions_pred_metrics(
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
        uncer_out = 0
        for i in range(uncer_step):
            uncer_out += sample_outputs[i]["model_outputs"][index]
        uncer_out = uncer_out / uncer_step
        uncer = compute_uncer(uncer_out).cpu()
        w = torch.exp(
            torch.sigmoid(torch.tensor((index + 1) / num_sample_timesteps))
            * (1 - uncer)
        )
        for i in range(uncer_step):
            sample_return += w * sample_outputs[i]["pred_xstarts"][index].cpu()

    return sample_return



def get_nonzero_patches(C, P, patch_labels, patch_data):
    '''
    
    Args:
        patch_labels: bool tensor (shape: (B, 1, W_, H_, D_))
        patch_data: float tensor (shape: (B, C*P^3, W_, H_, D_))
                data for every patch location
    Returns:
        nonzero_patches: float tensor (shape: (num_nonzero_patches, C, P, P, P))
    '''
    
    # Find nonzero patches
    nonzero_patch_indices = patch_labels.nonzero(as_tuple=True)
    num_nonzero_patches = nonzero_patch_indices[0].shape[0]
    
    # get the patch data corresponding to nonzero patch indices
    # B,C*P*P*P,W_,H_,D_ -> (num_nonzero_patches ,C*P*P*P)
    nonzero_patches = patch_data[
        nonzero_patch_indices[0],
        :,
        nonzero_patch_indices[2],
        nonzero_patch_indices[3],
        nonzero_patch_indices[4],
    ]

    # nonzero_patch: (num_nonzero_patches, C*P*P*P) -> (num_nonzero_patches, C, patch_size, patch_size, patch_size)
    nonzero_patches = nonzero_patches.view(
        -1, C, P, P, P
    )

    assert nonzero_patches.shape == (
        num_nonzero_patches,
        C,
        P,
        P,
        P,
    )
    
    return nonzero_patches


def fill_in_window_with_patches(window,patch_size,patch_labels,fill_patches):
    '''
    fill in a (zero tensor) window, at the tumor (non-tumor) patch locations [patch_labels]
    with respective tumor (non-tumor) patch data [fill_patches]

    Args:
        window: tensor for the entire window (shape: (B, C, W, H, D))
            window tensor should be uninitialized ie zero at locations where patch_labels==1
        patch_labels: bool tensor (shape: (B, 1, W_, H_, D_)) indexes the patch locations, (W_= W//P, H_= H//P, D_= D//P)
        fill_patches: float tensor (shape: (X, C, P, P, P)) , patch data to fill in with
                where X is the number of patches where patch_labels==1
    Returns:
        nonzero_patches: float tensor (shape: (num_nonzero_patches, C, P, P, P))
    '''
    
    # fill in the zero tensor pred_mask created before, at the tumor patch locations
    #  with the corresponding predicted mask patches
    fill_indices = patch_labels.nonzero(as_tuple=False)
    assert fill_patches.shape[0] == len(fill_indices)

    for j in range(len(fill_indices)):
        b, _, w_idx, h_idx, d_idx = fill_indices[j]
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



















