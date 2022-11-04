#%%
import os

import IPython.display as ipy_display
import ipywidgets
import matplotlib.pyplot as plt
import torch as th
import torch.utils.data as th_data
from experiments.datasets.mnist import ExpWAE

from gen_mods.common.autoencoders import Autoencoder

#%%
th.set_grad_enabled(False)

#%%
PROJECT_ROOT: str = "../../"
DATASET_ROOT: str = os.path.join(PROJECT_ROOT, "dataset")
CKPT_PATH: str = os.path.join(
    PROJECT_ROOT,
    "run/mnist_wae/train_local/ckpt/version_3/epoch=31-step=0.ckpt")

#%%
exp = ExpWAE.create_experiment(DATASET_ROOT)
ae = Autoencoder(enc=exp.enc, dec=exp.dec, img_proc=exp.img_proc).eval()
loader = th_data.DataLoader(
    dataset=exp.val_set if exp.val_set is not None else exp.train_set,
    batch_size=1,
    shuffle=True)

#%%
ckpt_f = th.load(f=CKPT_PATH, map_location=th.device("cpu"))
ae.load_state_dict(ckpt_f["state_dict"])

#%%
imgs, labels = next(iter(loader))
codes = ae.encode(imgs)
imgs_rec = ae.decode(codes)
plt.imshow(th.cat((imgs[0, 0], imgs_rec[0, 0]), dim=1), cmap="gray")

# %%
slides = [ipywidgets.FloatSlider(value=v, min=-50, max=50) for v in codes[0]]
button = ipywidgets.Button(description="Done")
ipy_display.display(*slides, button)


def _plot_img_callback(btn):
    codes_cf = th.as_tensor([v.value for v in slides])[None, :]
    imgs_cf = ae.decode(codes_cf)
    plt.imshow(th.cat((imgs[0, 0], imgs_cf[0, 0]), dim=1), cmap="gray")


button.on_click(_plot_img_callback)

#%%
