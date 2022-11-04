#%%
import os

import kornia as kn
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import PIL.Image
import torch as th
import torch.utils.data as th_data
import torchvision.transforms as thvis_tf
from torchvision.datasets import MNIST

#%%
PROJECT_ROOT: str = "../../"
DATASET_ROOT: str = os.path.join(PROJECT_ROOT, "dataset")
ONNX_ROOT: str = os.path.join(PROJECT_ROOT,
                              "run/mnist_wae/train_local/onnx/version_3/")

#%%
val_dataset = MNIST(root=DATASET_ROOT,
                    train=False,
                    download=True,
                    transform=thvis_tf.ToTensor())
val_loader = th_data.DataLoader(val_dataset, 1)

#%%
# CHW
imgs, labels = next(iter(val_loader))
plt.imshow(imgs[0, 0], cmap="gray")
print(imgs.shape)

# %%
encoder = ort.InferenceSession(os.path.join(ONNX_ROOT, "encoder.onnx"))
decoder = ort.InferenceSession(os.path.join(ONNX_ROOT, "decoder.onnx"))

#%%
codes_orig = encoder.run(output_names=["outputs"],
                         input_feed={"inputs": imgs.numpy()})[0]
imgs_rec_ort = decoder.run(output_names=["outputs"],
                           input_feed={"inputs": codes_orig})[0]
plt.imshow(imgs_rec_ort[0, 0], cmap="gray")


#%%
def load_image():
    with PIL.Image.open("7.jpg") as img_p:
        img_p = img_p.convert("RGB")
    img_np = np.array(img_p)
    img_np = img_np[None, :, :, :]
    img_np = np.transpose(img_np, (0, 3, 1, 2))
    return img_np


@th.no_grad()
def preprocess_image(imgs: np.ndarray) -> np.ndarray:
    """Transform an image into encoder input.

    Args:
        imgs (np.ndarray): (1, 3, h, w) input image.

    Returns:
        np.ndarray: (1, 1, h, w) the preprocessed image.
    """
    imgs_t: th.Tensor = th.as_tensor(imgs)
    # scale from [0, 255] to [0, 1]
    imgs_t = imgs_t / 255.0
    # cast to float32
    imgs_t = imgs_t.to(dtype=th.float32)
    # transform color
    imgs_t = kn.color.rgb_to_grayscale(imgs_t)
    # back to ndarray
    imgs = imgs_t.numpy()
    return imgs


#%%
imgs_l = load_image()
imgs_l = preprocess_image(imgs_l)
plt.imshow(imgs_l[0, 0], cmap="gray")

# %%
codes_l: np.ndarray = encoder.run(output_names=["outputs"],
                                  input_feed={"inputs": imgs_l})[0]
print(codes_orig)
print(codes_l)

#%%
imgs_rec_l = decoder.run(output_names=["outputs"],
                         input_feed={"inputs": codes_l})[0]
plt.imshow(imgs_rec_l[0, 0], cmap="gray")

#%%
