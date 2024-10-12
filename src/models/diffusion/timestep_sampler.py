from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


class ScheduleSampler(ABC):
    """A distribution over timesteps in the diffusion process, intended to reduce variance of the
    objective.

    By default, samplers perform unbiased importance sampling, in which the objective's mean is
    unchanged. However, subclasses may override sample() to change how the resampled terms are
    reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch):
        """Importance-sample timesteps for a batch. same as sample(self,batch_size, device), but,
        the signature is changed due to compaitibility with lightning as no T.to(device) calls
        allowed device is inferred from batch.

        :param batch: The batch for which the
                    timesteps have to be sampled.

        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        batch_size = batch.shape[0]
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).to(batch).long()
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).to(batch).float()
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion_steps):
        self.diffusion_steps = diffusion_steps
        self._weights = np.ones([diffusion_steps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the corresponding losses for
        each of those timesteps. This method will perform synchronization to make sure all of the
        ranks maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        if dist.is_available() and dist.is_initialized():
            batch_sizes = [
                th.tensor([0], dtype=th.int32).to(local_ts)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(
                batch_sizes,
                th.tensor([len(local_ts)], dtype=th.int32).to(local_ts),
            )
        else:
            batch_sizes = [th.tensor([len(local_ts)], dtype=th.int32).to(local_ts)]

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        if dist.is_available() and dist.is_initialized():
            timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
            loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
            dist.all_gather(timestep_batches, local_ts)
            dist.all_gather(loss_batches, local_losses)
        else:
            timestep_batches = [local_ts]
            loss_batches = [local_losses]

        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting using losses from the
        model.

        This method directly updates the reweighting without synchronizing between workers. It is
        called by update_with_local_losses from all ranks with identical arguments. Thus, it should
        have deterministic behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion_steps, history_per_term=10, uniform_prob=0.001):
        self.diffusion_steps = diffusion_steps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion_steps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion_steps], dtype=np.int64)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion_steps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
