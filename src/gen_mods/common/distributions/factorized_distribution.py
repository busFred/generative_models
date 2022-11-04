# https://github.com/pytorch/pytorch/pull/32377
from typing import List, Sequence

import torch
import torch.distributions as torch_dists

from . import utils


class FactorizedDistribution(torch_dists.Distribution):

    dists: Sequence[torch_dists.Distribution]
    secs: Sequence[int]

    def __init__(self, dists: Sequence[torch_dists.Distribution]):
        batch_shape, event_shape, self.secs = utils.check_batch_event_shape(dists)
        super().__init__(batch_shape, event_shape, validate_args=False)
        self.dists = dists

    def expand(self, batch_shape: torch.Size, _instance=None):
        dists_new: List[torch_dists.Distribution] = [
            dist.expand(batch_shape, _instance) for dist in self.dists
        ]
        return dists_new

    @property
    def mean(self):
        mean: torch.Tensor = torch.empty((*self.batch_shape, 0))
        for dist in self.dists:
            m: torch.Tensor = dist.mean
            if len(dist.event_shape) == 0:
                m = m[..., None]
            if mean.device != m.device:
                mean = mean.to(m.device)
            mean = torch.concat((mean, m), dim=-1)
        return mean

    @property
    def variance(self):
        variaince: torch.Tensor = torch.empty((*self.batch_shape, 0))
        for dist in self.dists:
            # dist.variance already discard covariance
            v: torch.Tensor = dist.variance
            if len(dist.event_shape) == 0:
                v = v[..., None]
            if variaince.device != v.device:
                variaince = variaince.to(v.device)
            variaince = torch.concat((variaince, v), dim=-1)
        return variaince

    def sample(self, sample_shape=torch.Size()):
        samps: torch.Tensor = torch.empty(
            (*sample_shape, *self.batch_shape, 0))
        for dist in self.dists:
            s: torch.Tensor = dist.sample(sample_shape)
            if len(dist.event_shape) == 0:
                s = s[..., None]
            if samps.device != s.device:
                samps = samps.to(s.device)
            samps = torch.concat((samps, s), dim=-1)
        return samps

    def rsample(self, sample_shape=torch.Size()):
        samps: torch.Tensor = torch.empty(
            (*sample_shape, *self.batch_shape, 0))
        for dist in self.dists:
            s: torch.Tensor = dist.rsample(sample_shape)
            if len(dist.event_shape) == 0:
                s = s[..., None]
            if samps.device != s.device:
                samps = samps.to(s.device)
            samps = torch.concat((samps, s), dim=-1)
        return samps

    # TODO could encounter device issues
    def log_prob(self, value: torch.Tensor):
        if self._validate_args:
            self._validate_sample(value)
        sample_shape = self._sample_shape(value)
        log_probs: torch.Tensor = torch.empty(
            (*sample_shape, *self.batch_shape, len(self.dists)),
            device=value.device)
        values: Sequence[torch.Tensor] = torch.split(value, self.secs, dim=-1)
        for idx, (dist, val) in enumerate(zip(self.dists, values)):
            if len(dist.event_shape) == 0:
                val = val.squeeze(dim=-1)
            log_probs[..., idx] = dist.log_prob(val)
        log_probs = log_probs.sum(dim=-1)
        return log_probs

    # TODO could encounter device issues
    def factorized_log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Factorized log likelihood evaluated at value

        Args:
            value (torch.Tensor): (*sample_shape, *batch_shape, *event_shape) The value to be evaluated.

        Returns:
            torch.Tensor: (*sample_shape, *batch_shape, *n_dists) log likelihood evaluated at value.
        """
        if self._validate_args:
            self._validate_sample(value)
        sample_shape = self._sample_shape(value)
        log_probs: torch.Tensor = torch.empty(
            (*sample_shape, *self.batch_shape, len(self.dists)),
            device=value.device)
        values: Sequence[torch.Tensor] = torch.split(value, self.secs, dim=-1)
        for idx, (dist, val) in enumerate(zip(self.dists, values)):
            if len(dist.event_shape) == 0:
                val = val.squeeze(dim=-1)
            log_probs[..., idx] = dist.log_prob(val)
        return log_probs

    # TODO could encounter device issue
    def cdf(self, value: torch.Tensor):
        self._validate_sample(value)
        sample_shape = self._sample_shape(value)
        cdfs: torch.Tensor = torch.empty(
            (*sample_shape, *self.batch_shape, len(self.dists)),
            device=value.device)
        values: Sequence[torch.Tensor] = torch.split(value, self.secs, dim=-1)
        for idx, (dist, val) in enumerate(zip(self.dists, values)):
            if len(dist.event_shape) == 0:
                val = val.squeeze(dim=-1)
            cdfs[..., idx] = dist.cdf(val)
        cdfs = cdfs.prod(dim=-1)
        return cdfs

    def entropy(self):
        entropy: torch.Tensor = torch.zeros(
            (*self.batch_shape, len(self.dists)))
        for idx, dist in enumerate(self.dists):
            ent: torch.Tensor = dist.entropy()
            if entropy.device != ent.device:
                entropy = entropy.to(ent.device)
            entropy[..., idx] = ent
        entropy = entropy.sum(dim=-1)
        return entropy

    # TODO could encounter device issues
    def factorized_entropy(self) -> torch.Tensor:
        """Factorized entropy.

        Returns:
            torch.Tensor: (*batch_shape, *n_dists) factorized entropy.
        """
        entropy: torch.Tensor = torch.zeros(
            (*self.batch_shape, len(self.dists)))
        for idx, dist in enumerate(self.dists):
            ent: torch.Tensor = dist.entropy()
            if entropy.device != ent.device:
                entropy = entropy.to(ent.device)
            entropy[..., idx] = ent
        return entropy

    def _sample_shape(self, value: torch.Tensor) -> torch.Size:
        sample_shape = value.shape[:-(len(self.batch_shape) +
                                      len(self.event_shape))]
        return sample_shape
