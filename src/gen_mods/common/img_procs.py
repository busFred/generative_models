from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import torch


class ImgProcBase(ABC):

    @abstractmethod
    def imgs_to_nnet_inputs(self, obss: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def nnet_outputs_to_imgs(self, nnet_outs: torch.Tensor) -> torch.Tensor:
        pass


class ImgProcEnsemble(ImgProcBase):

    img_procs: Sequence[ImgProcBase]

    def __init__(self, img_procs: Sequence[ImgProcBase]) -> None:
        super().__init__()
        self.img_procs = img_procs

    def imgs_to_nnet_inputs(self, obss: torch.Tensor) -> torch.Tensor:
        for img_proc in self.img_procs:
            obss = img_proc.imgs_to_nnet_inputs(obss)
        return obss

    def nnet_outputs_to_imgs(self, nnet_outs: torch.Tensor) -> torch.Tensor:
        for img_proc in self.img_procs:
            nnet_outs = img_proc.nnet_outputs_to_imgs(nnet_outs)
        return nnet_outs


class IdentityImgProc(ImgProcBase):

    def imgs_to_nnet_inputs(self, obss: torch.Tensor) -> torch.Tensor:
        return obss

    def nnet_outputs_to_imgs(self, nnet_outs: torch.Tensor) -> torch.Tensor:
        return nnet_outs


class ScaleImgProc(ImgProcBase):

    factor: float

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def imgs_to_nnet_inputs(self, obss: torch.Tensor) -> torch.Tensor:
        obss = obss / self.factor
        return obss

    def nnet_outputs_to_imgs(self, nnet_outs: torch.Tensor) -> torch.Tensor:
        nnet_outs = nnet_outs * self.factor
        return nnet_outs


class FrameSelectProc(ImgProcBase):

    def imgs_to_nnet_inputs(self, obss: torch.Tensor) -> torch.Tensor:
        obss = obss[:, -1]
        return obss

    def nnet_outputs_to_imgs(self, nnet_outs: torch.Tensor) -> torch.Tensor:
        return nnet_outs


class PermuteImgProc(ImgProcBase):

    dims: Tuple[int, ...]

    def __init__(self, dims: Tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def imgs_to_nnet_inputs(self, obss: torch.Tensor) -> torch.Tensor:
        obss = torch.permute(obss, self.dims)
        return obss

    def nnet_outputs_to_imgs(self, nnet_outs: torch.Tensor) -> torch.Tensor:
        dims = torch.argsort(torch.as_tensor(self.dims))
        nnet_outs = torch.permute(nnet_outs, dims.tolist())
        return nnet_outs
