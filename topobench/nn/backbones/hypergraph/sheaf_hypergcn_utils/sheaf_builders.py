"""Restriction-map (sheaf) builders for the SheafHyperGCN backbone.

Adapted from ``models/sheaf_utils/real_sheaf_builders.py`` and
``models/mlp.py`` of the reference Directional Sheaf Hypergraph Networks
implementation. For every incidence pair ``(v, e)`` these modules predict a
``d x d`` restriction map ``F_{v<e}`` (diagonal, orthogonal, or general).

The orthogonal builder reuses TopoBench's existing
:class:`topobench.nn.backbones.graph.nsd_utils.orthogonal.Orthogonal`
(``cayley`` / ``matrix_exp``), so no extra dependency is required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

from topobench.nn.backbones.graph.nsd_utils.orthogonal import Orthogonal


class MLP(nn.Module):
    """Multi-layer perceptron with optional input / hidden normalization.

    Adapted from https://github.com/CUAI/CorrectAndSmooth. With
    ``num_layers=1`` it reduces to a single linear layer (optionally preceded
    by a normalization of the input).

    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_channels : int
        Number of hidden features (used only when ``num_layers > 1``).
    out_channels : int
        Number of output features.
    num_layers : int
        Number of linear layers.
    dropout : float, optional
        Dropout probability applied between hidden layers (default: 0.5).
    Normalization : str, optional
        Normalization type, one of ``'bn'``, ``'ln'`` or ``'None'``
        (default: ``'bn'``).
    InputNorm : bool, optional
        Whether to normalize the input features (default: False).
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout=0.5,
        Normalization="bn",
        InputNorm=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ["bn", "ln", "None"]
        norm_cls = {
            "bn": nn.BatchNorm1d,
            "ln": nn.LayerNorm,
            "None": None,
        }[Normalization]

        input_norm = (
            norm_cls(in_channels)
            if (norm_cls is not None and InputNorm)
            else nn.Identity()
        )
        if num_layers == 1:
            self.normalizations.append(input_norm)
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.normalizations.append(input_norm)
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.normalizations.append(
                    norm_cls(hidden_channels)
                    if norm_cls is not None
                    else nn.Identity()
                )
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        """Reset all linear layers and normalization statistics."""
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if normalization.__class__.__name__ != "Identity":
                normalization.reset_parameters()

    def forward(self, x):
        """Apply the MLP to the input features.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``[*, in_channels]``.

        Returns
        -------
        torch.Tensor
            Output features of shape ``[*, out_channels]``.
        """
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


def _apply_act(h_sheaf, sheaf_act):
    """Apply the non-linearity used when predicting the sheaf blocks.

    Parameters
    ----------
    h_sheaf : torch.Tensor
        Predicted (pre-activation) block entries.
    sheaf_act : str
        Activation name, one of ``'sigmoid'``, ``'tanh'`` or ``'none'``.

    Returns
    -------
    torch.Tensor
        Activated block entries.
    """
    if sheaf_act == "sigmoid":
        return torch.sigmoid(h_sheaf)
    if sheaf_act == "tanh":
        return torch.tanh(h_sheaf)
    return h_sheaf


def predict_blocks(x, e, hyperedge_index, sheaf_lin, args):
    """Predict blocks as ``sigma(MLP(x_v || h_e))`` (hyperedge attr given).

    Parameters
    ----------
    x : torch.Tensor
        Node features of shape ``[num_nodes, f]``.
    e : torch.Tensor
        Hyperedge features of shape ``[num_edges, f]``.
    hyperedge_index : torch.Tensor
        Incidence index of shape ``[2, nnz]`` (row: nodes, col: hyperedges).
    sheaf_lin : MLP
        The MLP projecting the concatenated features to the block entries.
    args : types.SimpleNamespace
        Configuration namespace exposing ``sheaf_act``.

    Returns
    -------
    torch.Tensor
        Predicted block entries of shape ``[nnz, *]``.
    """
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)
    es = torch.index_select(e, dim=0, index=col)
    h_sheaf = torch.cat((xs, es), dim=-1)
    h_sheaf = sheaf_lin(h_sheaf)
    return _apply_act(h_sheaf, args.sheaf_act)


def predict_blocks_var2(x, hyperedge_index, sheaf_lin, args):
    """Predict blocks using the mean of node features as hyperedge attr.

    Parameters
    ----------
    x : torch.Tensor
        Node features of shape ``[num_nodes, f]``.
    hyperedge_index : torch.Tensor
        Incidence index of shape ``[2, nnz]`` (row: nodes, col: hyperedges).
    sheaf_lin : MLP
        The MLP projecting the concatenated features to the block entries.
    args : types.SimpleNamespace
        Configuration namespace exposing ``sheaf_act``.

    Returns
    -------
    torch.Tensor
        Predicted block entries of shape ``[nnz, *]``.
    """
    row, col = hyperedge_index
    e = scatter_mean(x[row], col, dim=0)
    xs = torch.index_select(x, dim=0, index=row)
    es = torch.index_select(e, dim=0, index=col)
    h_sheaf = torch.cat((xs, es), dim=-1)
    h_sheaf = sheaf_lin(h_sheaf)
    return _apply_act(h_sheaf, args.sheaf_act)


def predict_blocks_var3(x, hyperedge_index, sheaf_lin, sheaf_lin2, args):
    """Predict blocks using an EDHNN-style universal hyperedge aggregation.

    Parameters
    ----------
    x : torch.Tensor
        Node features of shape ``[num_nodes, f]``.
    hyperedge_index : torch.Tensor
        Incidence index of shape ``[2, nnz]`` (row: nodes, col: hyperedges).
    sheaf_lin : MLP
        The MLP projecting the concatenated features to the block entries.
    sheaf_lin2 : MLP
        The MLP applied to node features before summing them per hyperedge.
    args : types.SimpleNamespace
        Configuration namespace exposing ``sheaf_act``.

    Returns
    -------
    torch.Tensor
        Predicted block entries of shape ``[nnz, *]``.
    """
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)
    x_e = sheaf_lin2(x)
    e = scatter_add(x_e[row], col, dim=0)
    es = torch.index_select(e, dim=0, index=col)
    h_sheaf = torch.cat((xs, es), dim=-1)
    h_sheaf = sheaf_lin(h_sheaf)
    return _apply_act(h_sheaf, args.sheaf_act)


def _reduce_features(x, e, d):
    """Average the ``d`` stalk copies to obtain per-node / per-edge features.

    Parameters
    ----------
    x : torch.Tensor
        Node features of shape ``[num_nodes * d, f]``.
    e : torch.Tensor
        Hyperedge features of shape ``[num_edges * d, f]``.
    d : int
        Stalk dimension.

    Returns
    -------
    torch.Tensor
        Node features of shape ``[num_nodes, f]``.
    torch.Tensor
        Hyperedge features of shape ``[num_edges, f]``.
    """
    num_nodes = x.shape[0] // d
    num_edges = e.shape[0] // d
    x = x.view(num_nodes, d, x.shape[-1]).mean(1)
    e = e.view(num_edges, d, e.shape[-1]).mean(1)
    return x, e


class HGCNSheafBuilderDiag(nn.Module):
    """Predict diagonal ``d x d`` restriction maps for each incidence pair.

    Parameters
    ----------
    args : types.SimpleNamespace
        Configuration namespace with the sheaf hyperparameters
        (``d``, ``MLP_hidden``, ``sheaf_pred_block``, ``sheaf_dropout``,
        ``sheaf_special_head``, ``AllSet_input_norm``, ``dropout``,
        ``sheaf_act``).
    hidden_dim : int
        Feature width of the layer this builder is attached to. Overrides
        ``args.MLP_hidden`` for the incoming node features.
    """

    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head
        self.d = args.d
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm
        self.dropout = args.dropout

        if self.prediction_type == "MLP_var1":
            in_channels = self.MLP_hidden + args.MLP_hidden
        else:
            in_channels = 2 * self.MLP_hidden

        self.sheaf_lin = MLP(
            in_channels=in_channels,
            hidden_channels=args.MLP_hidden,
            out_channels=self.d,
            num_layers=1,
            dropout=0.0,
            Normalization="ln",
            InputNorm=self.norm,
        )
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin2 = MLP(
                in_channels=self.MLP_hidden,
                hidden_channels=args.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                Normalization="ln",
                InputNorm=self.norm,
            )

    def reset_parameters(self):
        """Reset the parameters of the internal MLP(s)."""
        self.sheaf_lin.reset_parameters()
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin2.reset_parameters()

    def forward(self, x, e, hyperedge_index):
        """Predict the diagonal block entries for each incidence pair.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes * d, f]``.
        e : torch.Tensor
            Hyperedge features of shape ``[num_edges * d, f]``.
        hyperedge_index : torch.Tensor
            Incidence index of shape ``[2, nnz]``.

        Returns
        -------
        torch.Tensor
            Diagonal entries of shape ``[nnz, d]``.
        """
        x, e = _reduce_features(x, e, self.d)

        if self.prediction_type == "MLP_var2":
            h_sheaf = predict_blocks_var2(
                x, hyperedge_index, self.sheaf_lin, self.args
            )
        elif self.prediction_type == "MLP_var3":
            h_sheaf = predict_blocks_var3(
                x, hyperedge_index, self.sheaf_lin, self.sheaf_lin2, self.args
            )
        else:
            h_sheaf = predict_blocks(
                x, e, hyperedge_index, self.sheaf_lin, self.args
            )

        if self.sheaf_dropout:
            h_sheaf = F.dropout(
                h_sheaf, p=self.dropout, training=self.training
            )
        return h_sheaf


class HGCNSheafBuilderGeneral(nn.Module):
    """Predict unconstrained ``d x d`` restriction maps per incidence pair.

    Parameters
    ----------
    args : types.SimpleNamespace
        Configuration namespace with the sheaf hyperparameters (see
        :class:`HGCNSheafBuilderDiag`).
    hidden_dim : int
        Feature width of the layer this builder is attached to.
    """

    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head
        self.d = args.d
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm
        self.dropout = args.dropout

        if self.prediction_type == "MLP_var1":
            in_channels = self.MLP_hidden + args.MLP_hidden
        else:
            in_channels = 2 * self.MLP_hidden

        self.sheaf_lin = MLP(
            in_channels=in_channels,
            hidden_channels=args.MLP_hidden,
            out_channels=self.d * self.d,
            num_layers=1,
            dropout=0.0,
            Normalization="ln",
            InputNorm=self.norm,
        )
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin2 = MLP(
                in_channels=self.MLP_hidden,
                hidden_channels=args.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                Normalization="ln",
                InputNorm=self.norm,
            )

    def reset_parameters(self):
        """Reset the parameters of the internal MLP(s)."""
        self.sheaf_lin.reset_parameters()
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin2.reset_parameters()

    def forward(self, x, e, hyperedge_index):
        """Predict the ``d*d`` block entries for each incidence pair.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes * d, f]``.
        e : torch.Tensor
            Hyperedge features of shape ``[num_edges * d, f]``.
        hyperedge_index : torch.Tensor
            Incidence index of shape ``[2, nnz]``.

        Returns
        -------
        torch.Tensor
            Block entries of shape ``[nnz, d * d]``.
        """
        x, e = _reduce_features(x, e, self.d)

        if self.prediction_type == "MLP_var2":
            h_sheaf = predict_blocks_var2(
                x, hyperedge_index, self.sheaf_lin, self.args
            )
        elif self.prediction_type == "MLP_var3":
            h_sheaf = predict_blocks_var3(
                x, hyperedge_index, self.sheaf_lin, self.sheaf_lin2, self.args
            )
        else:
            h_sheaf = predict_blocks(
                x, e, hyperedge_index, self.sheaf_lin, self.args
            )

        if self.sheaf_dropout:
            h_sheaf = F.dropout(
                h_sheaf, p=self.dropout, training=self.training
            )
        return h_sheaf


class HGCNSheafBuilderOrtho(nn.Module):
    """Predict orthogonal ``d x d`` restriction maps per incidence pair.

    The MLP predicts ``d * (d + 1) // 2`` parameters that are mapped to an
    orthogonal matrix through the Cayley / matrix-exponential transform (via
    :class:`topobench.nn.backbones.graph.nsd_utils.orthogonal.Orthogonal`).

    Parameters
    ----------
    args : types.SimpleNamespace
        Configuration namespace with the sheaf hyperparameters (see
        :class:`HGCNSheafBuilderDiag`) plus ``orthogonal_map``.
    hidden_dim : int
        Feature width of the layer this builder is attached to.
    """

    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head
        self.d = args.d
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm
        self.dropout = args.dropout

        self.orth_transform = Orthogonal(
            d=self.d, orthogonal_map=args.orthogonal_map
        )
        num_params = self.d * (self.d + 1) // 2

        if self.prediction_type == "MLP_var1":
            in_channels = self.MLP_hidden + args.MLP_hidden
        else:
            in_channels = 2 * self.MLP_hidden

        self.sheaf_lin = MLP(
            in_channels=in_channels,
            hidden_channels=args.MLP_hidden,
            out_channels=num_params,
            num_layers=1,
            dropout=0.0,
            Normalization="ln",
            InputNorm=self.norm,
        )
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin2 = MLP(
                in_channels=self.MLP_hidden,
                hidden_channels=args.MLP_hidden,
                out_channels=self.MLP_hidden,
                num_layers=1,
                dropout=0.0,
                Normalization="ln",
                InputNorm=self.norm,
            )

    def reset_parameters(self):
        """Reset the parameters of the internal MLP(s)."""
        self.sheaf_lin.reset_parameters()
        if self.prediction_type == "MLP_var3":
            self.sheaf_lin2.reset_parameters()

    def forward(self, x, e, hyperedge_index):
        """Predict the orthogonal ``d x d`` blocks for each incidence pair.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes * d, f]``.
        e : torch.Tensor
            Hyperedge features of shape ``[num_edges * d, f]``.
        hyperedge_index : torch.Tensor
            Incidence index of shape ``[2, nnz]``.

        Returns
        -------
        torch.Tensor
            Orthogonal blocks of shape ``[nnz, d, d]``.
        """
        x, e = _reduce_features(x, e, self.d)

        if self.prediction_type == "MLP_var2":
            h_sheaf = predict_blocks_var2(
                x, hyperedge_index, self.sheaf_lin, self.args
            )
        elif self.prediction_type == "MLP_var3":
            h_sheaf = predict_blocks_var3(
                x, hyperedge_index, self.sheaf_lin, self.sheaf_lin2, self.args
            )
        else:
            h_sheaf = predict_blocks(
                x, e, hyperedge_index, self.sheaf_lin, self.args
            )

        h_sheaf = self.orth_transform(h_sheaf)
        if self.sheaf_dropout:
            h_sheaf = F.dropout(
                h_sheaf, p=self.dropout, training=self.training
            )
        return h_sheaf
