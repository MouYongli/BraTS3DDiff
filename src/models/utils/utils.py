from operator import itemgetter

import torch.distributed as dist


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
    if dist.is_available():
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
