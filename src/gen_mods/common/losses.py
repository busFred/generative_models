import torch as th


def mmd_loss(z_tilde: th.Tensor, z: th.Tensor, z_var: float):
    r"""Calculate maximum mean discrepancy described in the WAE paper.
    
    Args:
        z_tilde (Tensor): samples from deterministic non-random encoder Q(Z|X).
            2D Tensor(batch_size x dimension).
        z (Tensor): samples from prior distributions. same shape with z_tilde.
        z_var (Number): scalar variance of isotropic gaussian prior P(Z).
    """
    assert z_tilde.size() == z.size()
    assert z.ndimension() == 2

    n = z.size(0)
    out = im_kernel_sum(z, z, z_var, exclude_diag=True).div_(n*(n-1)) + \
          im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div_(n*(n-1)) + \
          -im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div_(n*n).mul_(2)
    return out


def im_kernel_sum(z1: th.Tensor,
                  z2: th.Tensor,
                  z_var: float,
                  exclude_diag: bool = True):
    r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.

    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2 * z_dim * z_var

    # z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    # z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    # kernel_matrix = C / (1e-9 + C + (z11 - z22).pow_(2).sum(2))
    kernel_matrix = C / (1e-9 + C + th.cdist(z1, z2, 2)**2)
    kernel_sum = kernel_matrix.sum()

    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum
