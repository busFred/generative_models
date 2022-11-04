#%%
import os

import matplotlib.pyplot as plt
import torch.utils.data as th_data
import torchvision.transforms as thvis_tf
from torchvision.datasets import MNIST

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
# CHW
imgs, labels = next(iter(val_loader))
plt.imshow(imgs[0, 0], cmap="gray")
print(imgs.shape)

# %%
