#%%
import os

import matplotlib.pyplot as plt
import torch as th
import torch.utils.data as th_data
import torchvision.transforms as thvis_tf
from torchvision.datasets import MNIST
import torchinfo

#%%
PROJECT_ROOT: str = "../../"
DATASET_ROOT: str = os.path.join(PROJECT_ROOT, "dataset")

#%%
val_dataset = MNIST(root=DATASET_ROOT,
                    train=False,
                    download=True,
                    transform=thvis_tf.ToTensor())
val_loader = th_data.DataLoader(val_dataset, 16)

#%%
imgs: th.Tensor
labels: th.Tensor
imgs, labels = next(iter(val_loader))

#%%
enc = th.nn.Sequential(
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
)
torchinfo.summary(enc,
                  input_data=imgs,
                  batch_dim=0,
                  col_names=["input_size", "output_size"],
                  row_settings=["var_names"],
                  device=th.device("cpu"))

#%%
codes = enc(imgs)

# %%
dec = th.nn.Sequential(
    th.nn.Linear(in_features=8, out_features=7 * 7 * 1024),
    th.nn.Unflatten(dim=1, unflattened_size=(1024, 7, 7)),
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
        out_channels=256,
        kernel_size=(1, 1),
        stride=1,
    ),
)
torchinfo.summary(dec,
                  input_data=codes,
                  batch_dim=0,
                  col_names=["input_size", "output_size"],
                  row_settings=["var_names"],
                  device=th.device("cpu"))

# %%
gen_imgs = dec(codes)

#%%
