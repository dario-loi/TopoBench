"""SheafHyperGCN backbone for the TopoBench training framework.

SheafHyperGCN generalizes the non-linear HyperGCN Laplacian to cellular
sheaves. For every incidence pair ``(v, e)`` it predicts a ``d x d``
restriction map ``F_{v<e}``, reduces each hyperedge to a weighted graph
(supremum / infimum, optionally with mediators) in the sheaf space, assembles
a block sheaf Laplacian ``Delta`` on ``R^{Nd}``, normalizes it, and diffuses
node features with the operator ``I - Delta``.

Adapted from the reference implementation (``models/sheafhgcn.py``) of
Duta et al., "Sheaf Hypergraph Networks" (NeurIPS 2023),
https://arxiv.org/abs/2309.17116. The reference dataset-specific hidden-width
schedule and internal classifier head are replaced by a constant hidden width
plus a final projection to ``out_channels`` so that the TopoBench readout
performs the classification.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter_add, scatter_mean

from topobench.nn.backbones.hypergraph.sheaf_hypergcn_utils.laplacians import (
    SheafLaplacianDiag,
    SheafLaplacianGeneral,
    SheafLaplacianOrtho,
)
from topobench.nn.backbones.hypergraph.sheaf_hypergcn_utils.sheaf_builders import (
    MLP,
    HGCNSheafBuilderDiag,
    HGCNSheafBuilderGeneral,
    HGCNSheafBuilderOrtho,
)
from topobench.nn.backbones.hypergraph.sheaf_hypergcn_utils.utils import (
    HyperGraphConvolution,
    batched_sym_matrix_pow,
    sparse_diagonal,
)

SHEAF_BUILDERS = {
    "DiagSheafs": (HGCNSheafBuilderDiag, SheafLaplacianDiag),
    "OrthoSheafs": (HGCNSheafBuilderOrtho, SheafLaplacianOrtho),
    "GeneralSheafs": (HGCNSheafBuilderGeneral, SheafLaplacianGeneral),
}


class SheafHyperGCN(nn.Module):
    """Sheaf Hypergraph Convolutional Network backbone.

    Parameters
    ----------
    in_channels : int
        Number of input node feature channels (the feature-encoder width).
    hidden_channels : int
        Hidden feature width used inside every sheaf diffusion layer.
    out_channels : int
        Number of output node feature channels (should match the encoder
        width so the wrapper residual and the readout line up).
    n_layers : int, optional
        Number of sheaf diffusion layers (default: 3).
    d : int, optional
        Stalk dimension of the sheaf. ``d >= 1`` for diagonal sheaves and
        ``d > 1`` for orthogonal / general sheaves (default: 2).
    sheaf_type : str, optional
        Type of restriction maps: ``'DiagSheafs'``, ``'OrthoSheafs'`` or
        ``'GeneralSheafs'`` (default: ``'DiagSheafs'``).
    dropout : float, optional
        Dropout probability applied between diffusion layers (default: 0.5).
    init_hedge : str, optional
        Hyperedge feature initialization: ``'avg'`` (mean of member nodes) or
        ``'rand'`` (default: ``'avg'``).
    sheaf_normtype : str, optional
        Laplacian normalization: ``'degree_norm'``, ``'sym_degree_norm'``,
        ``'block_norm'`` or ``'sym_block_norm'`` (default: ``'sym_block_norm'``).
    sheaf_act : str, optional
        Non-linearity used when predicting the blocks: ``'sigmoid'``,
        ``'tanh'`` or ``'none'`` (default: ``'sigmoid'``).
    sheaf_left_proj : bool, optional
        Whether to left-multiply the features by a learned ``(I x W)`` map
        before each diffusion step (default: False).
    dynamic_sheaf : bool, optional
        Whether to rebuild the sheaf at every layer (default: False).
    sheaf_pred_block : str, optional
        How the hyperedge attribute is computed for block prediction:
        ``'MLP_var1'``, ``'MLP_var2'`` or ``'MLP_var3'`` (default:
        ``'MLP_var1'``).
    sheaf_dropout : bool, optional
        Whether to apply dropout to the predicted blocks (default: False).
    sheaf_special_head : bool, optional
        Whether to add a special identity head to the blocks (default: False).
    AllSet_input_norm : bool, optional
        Whether to normalize the input of the block-prediction MLPs
        (default: True).
    use_mediators : bool, optional
        Whether to connect supremum / infimum to mediator nodes when reducing
        each hyperedge (default: True).
    orthogonal_map : str, optional
        Parametrization used for orthogonal restriction maps: ``'cayley'`` or
        ``'matrix_exp'`` (default: ``'cayley'``).
    **kwargs : dict
        Additional keyword arguments (ignored).
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers=3,
        d=2,
        sheaf_type="DiagSheafs",
        dropout=0.5,
        init_hedge="avg",
        sheaf_normtype="sym_block_norm",
        sheaf_act="sigmoid",
        sheaf_left_proj=False,
        dynamic_sheaf=False,
        sheaf_pred_block="MLP_var1",
        sheaf_dropout=False,
        sheaf_special_head=False,
        AllSet_input_norm=True,
        use_mediators=True,
        orthogonal_map="cayley",
        **kwargs,
    ):
        super().__init__()

        if sheaf_type not in SHEAF_BUILDERS:
            raise ValueError(f"Unknown sheaf type: {sheaf_type}")
        if sheaf_type in ("OrthoSheafs", "GeneralSheafs"):
            assert d > 1, f"{sheaf_type} requires d > 1"

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.d = d
        self.n_layers = n_layers
        self.dropout = dropout
        self.init_hedge = init_hedge
        self.sheaf_normtype = sheaf_normtype
        self.left_proj = sheaf_left_proj
        self.dynamic_sheaf = dynamic_sheaf
        self.sheaf_type = sheaf_type
        self.use_mediators = use_mediators

        # Namespace consumed by the ported builder / Laplacian code.
        self._args = SimpleNamespace(
            d=d,
            MLP_hidden=hidden_channels,
            sheaf_pred_block=sheaf_pred_block,
            sheaf_dropout=sheaf_dropout,
            sheaf_special_head=sheaf_special_head,
            AllSet_input_norm=AllSet_input_norm,
            dropout=dropout,
            sheaf_act=sheaf_act,
            orthogonal_map=orthogonal_map,
            sheaf_normtype=sheaf_normtype,
        )

        model_sheaf, self.Laplacian = SHEAF_BUILDERS[sheaf_type]

        if self.left_proj:
            self.lin_left_proj = nn.ModuleList(
                [
                    MLP(
                        in_channels=d,
                        hidden_channels=d,
                        out_channels=d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization="ln",
                        InputNorm=AllSet_input_norm,
                    )
                    for _ in range(n_layers)
                ]
            )

        self.lin = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels * d,
            num_layers=1,
            dropout=0.0,
            Normalization="ln",
            InputNorm=False,
        )

        self.sheaf_builder = nn.ModuleList(
            [model_sheaf(self._args, hidden_channels)]
        )
        if self.dynamic_sheaf:
            for _ in range(1, n_layers):
                self.sheaf_builder.append(
                    model_sheaf(self._args, hidden_channels)
                )

        self.layers = nn.ModuleList(
            [
                HyperGraphConvolution(hidden_channels, hidden_channels)
                for _ in range(n_layers)
            ]
        )
        self.lin2 = nn.Linear(hidden_channels * d, out_channels, bias=False)

    def reset_parameters(self):
        """Reset the parameters of every learnable submodule."""
        for layer in self.layers:
            layer.reset_parameters()
        if self.left_proj:
            for lin_layer in self.lin_left_proj:
                lin_layer.reset_parameters()
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        for sheaf_builder in self.sheaf_builder:
            sheaf_builder.reset_parameters()

    def init_hyperedge_attr(self, num_edges, x, hyperedge_index):
        """Initialize hyperedge features from the node features.

        Parameters
        ----------
        num_edges : int
            Number of hyperedges.
        x : torch.Tensor
            Node features of shape ``[num_nodes, in_channels]``.
        hyperedge_index : torch.Tensor
            Incidence index of shape ``[2, nnz]``.

        Returns
        -------
        torch.Tensor
            Hyperedge features of shape ``[num_edges, in_channels]``.
        """
        if self.init_hedge == "rand":
            return torch.randn((num_edges, x.shape[-1]), device=x.device)
        return scatter_mean(x[hyperedge_index[0]], hyperedge_index[1], dim=0)

    def normalise(self, A, hyperedge_index, num_nodes, d):
        """Normalize the block sheaf Laplacian.

        The degree normalizations keep ``A`` sparse (the diagonal degree is
        applied as a differentiable scaling of the non-zero values). The block
        normalizations follow the reference and densify ``A`` (already
        required to extract the block-diagonal degree) and return a dense
        tensor.

        Parameters
        ----------
        A : torch.Tensor
            Sparse Laplacian of shape ``[num_nodes * d, num_nodes * d]``.
        hyperedge_index : torch.Tensor
            Laplacian indices used to compute the (block) degree.
        num_nodes : int
            Number of nodes.
        d : int
            Stalk dimension.

        Returns
        -------
        torch.Tensor
            The normalized Laplacian (sparse for degree norms, dense for block
            norms).
        """
        nd = num_nodes * d
        if self.sheaf_normtype in ("degree_norm", "sym_degree_norm"):
            deg = scatter_add(
                hyperedge_index.new_ones(hyperedge_index.size(1)).float(),
                hyperedge_index[0],
                dim=0,
                dim_size=nd,
            )
            A = A.coalesce()
            idx, val = A.indices(), A.values()
            if self.sheaf_normtype == "degree_norm":
                dinv = torch.pow(deg, -1.0)
                dinv[dinv == float("inf")] = 0
                new_val = val * dinv[idx[0]]
            else:
                dinv = torch.pow(deg, -0.5)
                dinv[dinv == float("inf")] = 0
                new_val = val * dinv[idx[0]] * dinv[idx[1]]
            A = torch.sparse_coo_tensor(idx, new_val, (nd, nd)).coalesce()

        elif self.sheaf_normtype in ("block_norm", "sym_block_norm"):
            power = -1.0 if self.sheaf_normtype == "block_norm" else -0.5
            A = A.to_dense()
            D = A.view((num_nodes, d, num_nodes, d))
            D = torch.permute(D, (0, 2, 1, 3))
            D = torch.diagonal(D, dim1=0, dim2=1)
            D = torch.permute(D, (2, 0, 1))

            if self.sheaf_type == "GeneralSheafs":
                D = batched_sym_matrix_pow(D, power)
            else:
                # A zero (isolated node) or negative (degenerate hyperedge
                # with supremum == infimum) diagonal entry makes the
                # elementwise power and its gradient blow up. Mask those
                # entries to zero and only raise the strictly-positive ones,
                # mirroring the spectral thresholding in the general case.
                good = D > 0
                D_safe = torch.where(good, D, torch.ones_like(D))
                D = torch.where(
                    good,
                    torch.pow(D_safe, power),
                    torch.zeros_like(D),
                )
            D = torch.block_diag(*torch.unbind(D, 0))

            A = D @ A
            if self.sheaf_normtype == "sym_block_norm":
                A = A @ D
            if self.sheaf_type == "GeneralSheafs":
                A = A.clamp(-1, 1)
        return A

    def forward(self, x, incidence):
        """Run the sheaf diffusion over the hypergraph.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes, in_channels]``.
        incidence : torch.Tensor
            Hypergraph incidence, either a sparse COO tensor of shape
            ``[num_nodes, num_hyperedges]`` or a dense ``[2, nnz]`` index.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``[num_nodes, out_channels]``.
        None
            Placeholder for the hyperedge embeddings (not produced).
        """
        if incidence.layout == torch.sparse_coo:
            edge_index, _ = torch_geometric.utils.to_edge_index(
                incidence.coalesce()
            )
        else:
            edge_index = incidence
        edge_index = edge_index.to(x.device)

        num_nodes = x.shape[0]
        num_edges = int(edge_index[1].max().item()) + 1
        d, m = self.d, self.use_mediators

        hyperedge_attr = self.init_hyperedge_attr(num_edges, x, edge_index)

        H = self.lin(x)
        hyperedge_attr = self.lin(hyperedge_attr)
        H = H.view((num_nodes * d, self.hidden_channels))
        hyperedge_attr = hyperedge_attr.view(
            (num_edges * d, self.hidden_channels)
        )

        A = None
        for i, layer in enumerate(self.layers):
            if i == 0 or self.dynamic_sheaf:
                sheaf = self.sheaf_builder[i](H, hyperedge_attr, edge_index)
                h_index, h_attr = self.Laplacian(H, m, d, edge_index, sheaf)
                A = torch.sparse_coo_tensor(
                    h_index, h_attr, (num_nodes * d, num_nodes * d)
                ).coalesce()
                A = self.normalise(A, h_index, num_nodes, d)
                # Diffusion operator I - Delta (keep A's sparse/dense layout).
                if A.is_sparse:
                    eye = torch.ones((num_nodes * d), device=x.device)
                    A = (
                        sparse_diagonal(
                            eye, (num_nodes * d, num_nodes * d)
                        ).to(A.device)
                        - A
                    )
                else:
                    A = torch.eye(num_nodes * d, device=x.device) - A

            if self.left_proj:
                H = H.t().reshape(-1, d)
                H = self.lin_left_proj[i](H)
                H = H.reshape(-1, num_nodes * d).t()

            H = F.relu(layer(A, H))
            if i < self.n_layers - 1:
                H = F.dropout(H, self.dropout, training=self.training)

        H = H.view(num_nodes, -1)
        H = self.lin2(H)
        return H, None
