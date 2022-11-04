import os
import sys
from argparse import ArgumentParser, Namespace
from typing import List

import pytorch_lightning.callbacks as pl_callbacks
import torch as th
import torch.utils.data as th_data
from experiments.datasets.mnist import ExpWAE, TFBLogger
from gen_mods.common.callbacks import ForceModelCheckpoint
from gen_mods.core.wae import WAE


def parse_args(args: List[str]) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_mmd", type=float, default=1.0)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--name", type=str, default="train_wae")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    argv = parser.parse_args(args)
    return argv


def main(args: List[str]):
    argv = parse_args(args)
    exp = ExpWAE.create_experiment(root=argv.dataset_root)
    train_loader = th_data.DataLoader(exp.train_set,
                                      batch_size=argv.batch_size,
                                      collate_fn=exp.collate_fn,
                                      num_workers=argv.n_workers,
                                      shuffle=True)
    val_loader = th_data.DataLoader(
        exp.val_set,
        batch_size=argv.batch_size,
        collate_fn=exp.collate_fn,
        num_workers=argv.n_workers,
        shuffle=False) if exp.val_set is not None else None
    wae = WAE(
        enc=exp.enc,
        dec=exp.dec,
        img_proc=exp.img_proc,
        prior=exp.latent_prior,
        optim_config=(th.optim.Adam, {
            "lr": argv.lr
        }),
    )
    logger = TFBLogger(
        save_dir=os.path.join(argv.output_dir, argv.name, "tfb"),
        name="",
        log_image_every_n_steps=10,
    )
    callbacks: List[pl_callbacks.Callback] = [
        ForceModelCheckpoint(
            dirpath=os.path.join(argv.output_dir, argv.name, "ckpt",
                                 f"version_{logger.version}"),
            every_n_epochs=1,
            save_top_k=-1,
        )
    ]
    wae.fit(train_loader=train_loader,
            n_iter=argv.n_iter,
            logger=logger,
            callbacks=callbacks,
            val_loader=val_loader,
            accelerator=argv.device)


if __name__ == "__main__":
    main(sys.argv[1:])
