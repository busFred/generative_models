#%%
import torch as th


#%%
class View(th.nn.Module):

    def __init__(self, dim, shape):
        super().__init__()
        self.dim = dim
        self.shape = shape

    def forward(self, input):
        new_shape = list(input.shape)[:self.dim] + list(self.shape) + list(
            input.shape)[self.dim + 1:]
        return input.view(*new_shape)


#%%
inputs = th.randn(8, 1024 * 7 * 7)

#%%
unflatten_outs = th.nn.Unflatten(1, (1024, 7, 7))(inputs)
view_outs = View(1, (1024, 7, 7))(inputs)
assert (unflatten_outs == view_outs).all()
