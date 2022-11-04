import itertools
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import pytorch_lightning.accelerators as pl_accelerators
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.plugins as pl_plugins
import pytorch_lightning.strategies as pl_strategies
import torch as th
import torch.distributions as th_dists
import torch.utils.data as th_data

from ..common.autoencoders import Autoencoder
from ..common.generators import Generator
from ..common.img_procs import ImgProcBase
from ..common.loggers import ITrainPredImageLogger
from ..common.losses import mmd_loss
from ..utils import train_utils


class WAE(Autoencoder, pl.LightningModule):

    prior: th_dists.Distribution

    lambda_rec: float
    lambda_mmd: float
    optim_config: Tuple[Type[th.optim.Optimizer], Dict[str, Any]]

    _autoenc_steps: int

    def __init__(
        self,
        enc: Generator,
        dec: Generator,
        img_proc: ImgProcBase,
        prior: th_dists.Distribution,
        lambda_rec: float = 1.0,
        lambda_mmd: float = 1.0,
        optim_config: Tuple[Type[th.optim.Optimizer],
                            Dict[str, Any]] = (th.optim.Adam, {
                                "lr": 1e-3
                            })
    ) -> None:
        super().__init__(enc, dec, img_proc)
        self.prior = prior
        # configure loss function
        self.lambda_rec = lambda_rec
        self.lambda_mmd = lambda_mmd
        # configure optimizer
        self.optim_config = optim_config
        # step counter
        self._autoenc_steps = 0
        # pl flags
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["enc", "dec"])

    def fit(self,
            train_loader: th_data.DataLoader,
            n_iter: int = 1000,
            logger: Union[pl_loggers.Logger, Iterable[pl_loggers.Logger],
                          bool] = True,
            callbacks: Optional[Union[List[pl_callbacks.Callback],
                                      pl_callbacks.Callback]] = None,
            val_loader: Optional[th_data.DataLoader] = None,
            ckpt_path: Optional[str] = None,
            accelerator: Optional[Union[str,
                                        pl_accelerators.Accelerator]] = None,
            devices: Optional[Union[List[int], str, int]] = None,
            strategy: Optional[Union[str, pl_strategies.Strategy]] = None,
            plugins: Optional[Union[pl_plugins.PLUGIN_INPUT,
                                    List[pl_plugins.PLUGIN_INPUT]]] = None):
        trainer = pl.Trainer(logger=logger,
                             callbacks=callbacks,
                             max_epochs=n_iter,
                             accelerator=accelerator,
                             devices=devices,
                             strategy=strategy,
                             plugins=plugins)
        trainer.fit(self, train_loader, val_loader, ckpt_path=ckpt_path)

    # pl train override
    def configure_optimizers(self):
        optim = self.optim_config[0](params=itertools.chain(
            self.enc.parameters(), self.dec.parameters()),
                                     **self.optim_config[1])
        return optim

    def training_step(self, batch: th.Tensor, batch_idx: int) -> None:
        # get optimizer
        optim: th.optim.Optimizer
        optim = self.optimizers(use_pl_optimizer=False)  # type: ignore
        # forward propagate ae
        nnet_ins: th.Tensor = self.img_proc.imgs_to_nnet_inputs(batch)
        codes: th.Tensor = self.enc(inputs=nnet_ins)
        nnet_outs: th.Tensor = self.dec(inputs=codes)
        preds: th.Tensor = self.img_proc.nnet_outputs_to_imgs(nnet_outs)
        # compute loss
        loss, metrics = self._compute_loss(nnet_ins=nnet_ins,
                                           codes=codes,
                                           nnet_outs=nnet_outs)
        # update module
        optim.zero_grad()
        loss.backward()
        optim.step()
        # log metrics
        metrics = train_utils.add_prefix_to_metrics(metrics, prefix="train")
        for logger in self.loggers:
            logger.log_metrics(metrics, step=self._autoenc_steps)
            if isinstance(logger, ITrainPredImageLogger):
                logger.log_image(name="train/recon_imgs",
                                 preds=preds,
                                 targets=batch,
                                 step=self._autoenc_steps)
        # increment steps
        self._autoenc_steps = self._autoenc_steps + 1

    # pl val override
    def validation_step(self, batch: th.Tensor,
                        batch_idx: int) -> Dict[str, float]:
        # forward propagate ae
        nnet_ins: th.Tensor = self.img_proc.imgs_to_nnet_inputs(batch)
        codes: th.Tensor = self.enc(inputs=nnet_ins)
        nnet_outs: th.Tensor = self.dec(inputs=codes)
        # compute loss
        _, metrics = self._compute_loss(nnet_ins=nnet_ins,
                                        codes=codes,
                                        nnet_outs=nnet_outs)
        return metrics

    def validation_epoch_end(self, outputs: List[Dict[str, float]]) -> None:
        keys = outputs[0].keys()
        metrics: Dict[str, float] = dict()
        for k in keys:
            metrics[k] = th.tensor([o[k] for o in outputs]).mean().item()
        metrics = train_utils.add_prefix_to_metrics(metrics, "val")
        for logger in self.loggers:
            logger.log_metrics(metrics, self.current_epoch)
        # log so that ray tune can do its job
        self.log_dict(metrics, logger=False)

    def _compute_loss(
            self, nnet_ins: th.Tensor, codes: th.Tensor,
            nnet_outs: th.Tensor) -> Tuple[th.Tensor, Dict[str, float]]:
        batch_size: int = nnet_ins.shape[0]
        l_rec: th.Tensor = th.nn.functional.mse_loss(input=nnet_outs,
                                                     target=nnet_ins)
        codes_p: th.Tensor = self.prior.sample(th.Size((batch_size, )))
        codes_p = codes_p.to(device=self.device)
        l_mmd: th.Tensor = mmd_loss(z_tilde=codes,
                                    z=codes_p,
                                    z_var=self.prior.variance.sum())
        loss: th.Tensor = self.lambda_rec * l_rec + self.lambda_mmd * l_mmd
        metrics = {
            "l_rec": l_rec.item(),
            "l_mmd": l_mmd.item(),
            "loss": loss.item()
        }
        return loss, metrics

    # other pl override
    @property
    def device(self) -> th.device:
        device = super().device
        if isinstance(device, th.device):
            return device
        return th.device(device)
