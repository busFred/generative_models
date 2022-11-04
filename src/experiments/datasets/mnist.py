from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Type, Union

import pytorch_lightning.loggers as pl_loggers
import torch as th
import torch.distributions as th_dists
import torch.utils.data as th_data
import torchvision.transforms as thvis
from gen_mods.common.generators import Generator
from gen_mods.common.img_procs import IdentityImgProc, ImgProcBase
from gen_mods.common.loggers import ITrainPredImageLogger
from gen_mods.nn.flatten import Unflatten
from torchvision.datasets import MNIST


def collate_fn(batch) -> th.Tensor:
    batch = th_data.default_collate(batch)
    return batch[0]


@dataclass
class ExpWAE:

    train_set: th_data.Dataset

    enc: Generator
    dec: Generator

    img_proc: ImgProcBase

    latent_prior: th_dists.Distribution

    val_set: Optional[th_data.Dataset] = None
    collate_fn: Callable = th_data.default_collate

    @classmethod
    def create_experiment(cls: Type["ExpWAE"], root: str) -> "ExpWAE":
        enc = Generator(gen_net=th.nn.Sequential(
            th.nn.Conv2d(
                in_channels=1,
                out_channels=128,
                kernel_size=(4, 4),
            ),
            th.nn.BatchNorm2d(num_features=128),
            th.nn.ReLU(inplace=True),
            th.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(4, 4),
            ),
            th.nn.BatchNorm2d(num_features=256),
            th.nn.ReLU(inplace=True),
            th.nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(4, 4),
            ),
            th.nn.BatchNorm2d(num_features=512),
            th.nn.ReLU(inplace=True),
            th.nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=(4, 4),
            ),
            th.nn.BatchNorm2d(num_features=1024),
            th.nn.ReLU(inplace=True),
            th.nn.Flatten(),
            th.nn.LazyLinear(out_features=8),
        ))
        dec = Generator(gen_net=th.nn.Sequential(
            th.nn.Linear(in_features=8, out_features=7 * 7 * 1024),
            Unflatten(dim=1, unflattened_size=(1024, 7, 7)),
            th.nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
            ),
            th.nn.BatchNorm2d(num_features=512),
            th.nn.ReLU(inplace=True),
            th.nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
            ),
            th.nn.BatchNorm2d(num_features=256),
            th.nn.ReLU(inplace=True),
            th.nn.ConvTranspose2d(
                in_channels=256,
                out_channels=1,
                kernel_size=(1, 1),
                stride=1,
            ),
            th.nn.Sigmoid(),
        ))
        img_proc = IdentityImgProc()
        train_set = MNIST(root=root,
                          train=True,
                          download=True,
                          transform=thvis.transforms.ToTensor())
        val_set = MNIST(root=root,
                        train=False,
                        download=True,
                        transform=thvis.transforms.ToTensor())
        latent_prior = th_dists.MultivariateNormal(loc=th.zeros(8),
                                                   covariance_matrix=th.eye(8))
        experiment = cls(train_set=train_set,
                         enc=enc,
                         dec=dec,
                         img_proc=img_proc,
                         latent_prior=latent_prior,
                         val_set=val_set,
                         collate_fn=collate_fn)
        return experiment


class TFBLogger(pl_loggers.TensorBoardLogger, ITrainPredImageLogger):
    log_image_every_n_steps: int

    def __init__(self,
                 save_dir: str,
                 name: Optional[str] = "lightning_logs",
                 version: Optional[Union[int, str]] = None,
                 log_graph: bool = False,
                 default_hp_metric: bool = True,
                 prefix: str = "",
                 sub_dir: Optional[str] = None,
                 agg_key_funcs: Optional[Mapping[str,
                                                 Callable[[Sequence[float]],
                                                          float]]] = None,
                 agg_default_func: Optional[Callable[[Sequence[float]],
                                                     float]] = None,
                 log_image_every_n_steps: int = 100,
                 **kwargs):
        super().__init__(save_dir, name, version, log_graph, default_hp_metric,
                         prefix, sub_dir, agg_key_funcs, agg_default_func,
                         **kwargs)
        self.log_image_every_n_steps = log_image_every_n_steps

    def log_image(self, name: str, preds: th.Tensor, targets: th.Tensor,
                  step: int):
        if step % self.log_image_every_n_steps != 0:
            return
        # preds and targets side by side
        imgs = th.cat((targets, preds), dim=3)
        self.experiment.add_images(tag=name,
                                   img_tensor=imgs,
                                   global_step=step,
                                   dataformats="NCHW")
