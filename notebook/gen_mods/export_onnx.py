#%%
import os

import numpy as np
import onnxruntime as ort
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
ONNX_ROOT: str = os.path.join(PROJECT_ROOT,
                              "run/mnist_wae/train_local/onnx/version_3/")

#%%
exp = ExpWAE.create_experiment(DATASET_ROOT)
ae = Autoencoder(enc=exp.enc, dec=exp.dec, img_proc=exp.img_proc).eval()
ckpt_f = th.load(f=CKPT_PATH, map_location=th.device("cpu"))
ae.load_state_dict(ckpt_f["state_dict"])

#%%
loader = th_data.DataLoader(
    dataset=exp.val_set if exp.val_set is not None else exp.train_set,
    batch_size=1,
    shuffle=True)
imgs, labels = next(iter(loader))

#%%
codes = ae.encode(imgs)
imgs_rec = ae.decode(codes)

#%%
ae.to_onnx(ONNX_ROOT, imgs)

#%%
encoder = ort.InferenceSession(os.path.join(ONNX_ROOT, "encoder.onnx"))
decoder = ort.InferenceSession(os.path.join(ONNX_ROOT, "decoder.onnx"))

#%%
codes_ort = encoder.run(output_names=["outputs"],
                        input_feed={"inputs": imgs.numpy()})[0]
np.testing.assert_allclose(codes_ort, codes.numpy(), rtol=1e-3, atol=1e-5)
imgs_rec_ort = decoder.run(output_names=["outputs"],
                           input_feed={"inputs": codes_ort})[0]
np.testing.assert_allclose(imgs_rec_ort,
                           imgs_rec.numpy(),
                           rtol=1e-3,
                           atol=1e-5)

# %%
