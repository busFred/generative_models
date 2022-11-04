import torch as th


class Unflatten(th.nn.Module):

    def __init__(self, dim, unflattened_size):
        super().__init__()
        self.dim = dim
        self.shape = unflattened_size

    def forward(self, input):
        new_shape = list(input.shape)[:self.dim] + list(self.shape) + list(
            input.shape)[self.dim + 1:]
        return input.view(*new_shape)
