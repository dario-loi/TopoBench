"""Low-level building blocks for the SheafHyperGCN backbone.

Adapted from the reference implementation of Directional Sheaf Hypergraph
Networks (``models/sheaf_utils/utils.py``). Only the pieces used by
``SheafHyperGCN`` are kept: a sparse graph-convolution layer, an autograd
sparse-dense matrix multiplication, and two sparse-matrix helpers.
"""

import math

import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class SparseMM(torch.autograd.Function):
    """Sparse x dense matrix multiplication with autograd support.

    Implementation by Soumith Chintala, see
    https://discuss.pytorch.org/t/does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    @staticmethod
    def forward(ctx, matrix_1, matrix_2):
        """Compute the product of a sparse and a dense matrix.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context object used to stash information for the backward pass.
        matrix_1 : torch.Tensor
            Sparse matrix of shape ``[n, m]``.
        matrix_2 : torch.Tensor
            Dense matrix of shape ``[m, k]``.

        Returns
        -------
        torch.Tensor
            The dense product of shape ``[n, k]``.
        """
        ctx.save_for_backward(matrix_1, matrix_2)
        return torch.mm(matrix_1, matrix_2)

    @staticmethod
    def backward(ctx, grad_output):
        """Backpropagate the gradient through the matrix product.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context object holding the tensors saved during the forward pass.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the layer output.

        Returns
        -------
        torch.Tensor or None
            Gradient with respect to the first (sparse) matrix, or None.
        torch.Tensor or None
            Gradient with respect to the second (dense) matrix, or None.
        """
        matrix_1, matrix_2 = ctx.saved_tensors
        grad_1 = grad_2 = None

        if ctx.needs_input_grad[0]:
            grad_1 = torch.mm(grad_output, matrix_2.t())

        if ctx.needs_input_grad[1]:
            grad_2 = torch.mm(matrix_1.t(), grad_output)

        return grad_1, grad_2


class HyperGraphConvolution(Module):
    """Sparse graph convolution layer, similar to Kipf and Welling (2017).

    Applies ``structure @ (H @ W) + b`` where ``structure`` is a precomputed
    sparse propagation operator (here the normalized sheaf Laplacian based
    ``I - Delta``).

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.W = Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the layer weights and bias with a uniform distribution."""
        std = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, structure, H):
        """Propagate features through the sparse structure.

        Parameters
        ----------
        structure : torch.Tensor
            Sparse propagation operator of shape ``[Nd, Nd]``.
        H : torch.Tensor
            Dense feature matrix of shape ``[Nd, in_channels]``.

        Returns
        -------
        torch.Tensor
            Propagated features of shape ``[Nd, out_channels]``.
        """
        HW = torch.mm(H, self.W)
        structure = structure.to(H.device)
        AHW = SparseMM.apply(structure, HW)
        return AHW + self.bias

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} "
            f"({self.in_channels} -> {self.out_channels})"
        )


def sparse_diagonal(diag, shape):
    """Build a sparse diagonal matrix from a dense diagonal vector.

    Parameters
    ----------
    diag : torch.Tensor
        Dense vector holding the diagonal entries.
    shape : tuple of int
        Shape ``(r, c)`` of the resulting square matrix (``r == c``).

    Returns
    -------
    torch.Tensor
        Sparse COO matrix with ``diag`` on its main diagonal.
    """
    r, c = shape
    assert r == c
    indexes = torch.arange(r, device=diag.device)
    indexes = torch.stack([indexes, indexes], dim=0)
    return torch.sparse_coo_tensor(indexes, diag, (r, c))


def batched_sym_matrix_pow(matrices: torch.Tensor, p: float) -> torch.Tensor:
    """Raise a batch of symmetric (Hermitian) matrices to a real power.

    Uses a singular value / eigen decomposition and applies the exponent to
    the (thresholded) spectrum.

    Parameters
    ----------
    matrices : torch.Tensor
        Tensor of shape ``(..., N, N)`` of symmetric or Hermitian matrices.
    p : float
        Exponent to raise each matrix to.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape, each matrix raised to the ``p``-th power.
    """
    if torch.is_complex(matrices):
        U, S, _ = torch.linalg.svd(matrices)
        max_vals, _ = S.max(-1, keepdim=True)
        tol = max_vals * S.size(-1) * torch.finfo(S.real.dtype).eps
        good = tol < S
        S_p = S.pow(p).where(good, torch.zeros_like(S))
        return (U * S_p.unsqueeze(-2)) @ U.conj().transpose(-2, -1)

    vecs, vals, _ = torch.linalg.svd(matrices)
    max_vals, _ = vals.max(-1, keepdim=True)
    tol = max_vals * vals.size(-1) * torch.finfo(vals.dtype).eps
    good = tol < vals
    vals_p = vals.pow(p).where(good, torch.zeros_like(vals))
    return (vecs * vals_p.unsqueeze(-2)) @ vecs.transpose(-2, -1)
