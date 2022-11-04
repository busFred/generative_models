from abc import ABC, abstractmethod
from typing import Tuple

import torch as th
import torch.distributions as th_dists


class Generator(th.nn.Module):

    gen_net: th.nn.Module

    def __init__(self, gen_net: th.nn.Module) -> None:
        """Constructor for `Generator`.
        Args:
            gen_net (th.nn.Module): the generating network that ouputs encoded values.
        """
        super().__init__()
        self.gen_net = gen_net

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        """Given an input, outputs encoded/decoded valeus deterministically.
        Args:
            inputs (th.Tensor): (n_batches, *in_features_shape) the inputs used for generation.
        Returns:
            th.Tensor: (n_batches, *out_features_shape) the generated outputs.
        """
        gen_outs = self.gen_net(inputs)
        codes = self.gen_outs_to_codes(gen_outs)
        return codes

    def gen_outs_to_codes(self, gen_outs: th.Tensor) -> th.Tensor:
        """Transform `gen_net` outputs into encoded/decoded values deterministically.
        Args:
            gen_outs (th.Tensor): (n_batch, *out_latents_shape) output produced by `gen_net`
        Returns:
            th.Tensor: (n_batch, *out_features_shape) the deterministic encoded values.
        """
        return gen_outs

    def _init_run(self, inputs: th.Tensor) -> None:
        self.forward(inputs)


class StochasticGeneratorBase(Generator, ABC):

    def __init__(self, gen_net: th.nn.Module) -> None:
        """Constructor for `StochasticGeneratorBase`.
        Args:
            gen_net (th.nn.Module): the generating network that ouputs encoded values AND the values used to produce generative distribution.
        """
        super().__init__(gen_net)

    @abstractmethod
    def gen_outs_to_dists(self, gen_outs: th.Tensor) -> th_dists.Distribution:
        """Transform `gen_net` outputs into generative distribution.
        Args:
            gen_outs (th.Tensor): (n_batch, *out_latents_shape) output produced by `gen_net`
        Returns:
            th_dists.Distribution: the generative distribution given inputs.
        """
        pass

    def compute_outs_codes_dists(
        self,
        inputs: th.Tensor,
        is_deterministic: bool = True
    ) -> Tuple[th.Tensor, th.Tensor, th_dists.Distribution]:
        gen_outs: th.Tensor = self.gen_net(inputs)
        out_dists: th_dists.Distribution = self.gen_outs_to_dists(gen_outs)
        codes: th.Tensor
        if is_deterministic:
            codes: th.Tensor = self.gen_outs_to_codes(gen_outs)
        else:
            codes: th.Tensor = out_dists.sample()
        return gen_outs, codes, out_dists

    def forward(self,
                inputs: th.Tensor,
                is_deterministic: bool = True) -> th.Tensor:
        """Given inputs, generate encoded/decoded values.
        Args:
            inputs (th.Tensor): (n_batches, *in_features_shape) the inputs used for generation.
            is_deterministic (bool, optional): wheather to generate deterministically. Defaults to True.
        Returns:
            th.Tensor: (n_batches, *out_features_shape) the generated outputs.
        """
        gen_outs: th.Tensor = self.gen_net(inputs)
        if is_deterministic:
            codes: th.Tensor = self.gen_outs_to_codes(gen_outs)
            return codes
        out_dists: th_dists.Distribution = self.gen_outs_to_dists(gen_outs)
        codes: th.Tensor = out_dists.sample()
        return codes

    def _init_run(self, inputs: th.Tensor) -> None:
        self.forward(inputs, True)
        self.forward(inputs, False)