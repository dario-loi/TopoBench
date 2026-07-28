"""Non-linear sheaf Laplacian builders for the SheafHyperGCN backbone.

Adapted from ``models/sheaf_utils/hgcn_sheaflaplacians.py`` of the reference
Directional Sheaf Hypergraph Networks implementation. Each hyperedge is
reduced to a weighted graph connecting the supremum / infimum nodes (and,
optionally, mediators) as in HyperGCN -- except the distances are measured in
the sheaf (opinion) space ``F_{v<e}(x_v)`` rather than the input space. The
resulting block sheaf Laplacian lives on ``R^{Nd}``.
"""

import itertools

import numpy as np
import torch
from torch_scatter import scatter_add


def update(supremum, infimum, mediator, weights, c):
    """Update the edge weights connecting a mediator to sup / inf nodes.

    Parameters
    ----------
    supremum : int
        Supremum node of the hyperedge.
    infimum : int
        Infimum node of the hyperedge.
    mediator : int
        Mediator node of the hyperedge.
    weights : dict
        Mapping from an ordered node pair to its accumulated weight.
    c : float
        Normalization constant ``2 * |e| - 3``.

    Returns
    -------
    dict
        The updated ``weights`` dictionary.
    """
    for pair in [
        (supremum, mediator),
        (infimum, mediator),
        (mediator, supremum),
        (mediator, infimum),
    ]:
        weights[pair] = weights.get(pair, 0) + float(1 / c)
    return weights


def reduce_graph(x_reduced, m, d, edge_index):
    """Reduce each hyperedge to a weighted graph in sheaf space.

    For every hyperedge the two nodes with maximum distance (in the sheaf
    space ``F_{v<e}(x_v)``) are connected; with mediators every remaining node
    is also connected to both. Self-loops are recorded so the diagonal
    ``sum_e F_{v<e}^T F_{v<e}`` can be aggregated later.

    Parameters
    ----------
    x_reduced : torch.Tensor
        Sheaf-transformed features of shape ``[nnz, d, f]``.
    m : bool
        Whether to connect supremum / infimum to the mediator nodes.
    d : int
        Stalk dimension.
    edge_index : torch.Tensor
        Incidence index of shape ``[2, nnz]`` (row: nodes, col: hyperedges).

    Returns
    -------
    torch.Tensor
        ``edges_idx``: indices into the ``nnz`` axis for the off-diagonal
        graph edges, of shape ``[2, K]``.
    torch.Tensor
        ``edges_idx_diag``: indices into the ``nnz`` axis for the diagonal
        (self-loop) contributions, of shape ``[2, D]``.
    torch.Tensor
        ``all_contained_hyperedges``: scatter grouping used to aggregate the
        diagonal contributions per node.
    torch.Tensor
        ``hgcn_edges``: the new graph edges (node ids) of shape ``[2, E]``.
    """
    weights = {}
    rv = torch.rand(x_reduced.shape[-1]).to(x_reduced.device).unsqueeze(-1)

    row, col = edge_index

    receivers_idx = torch.arange(len(row), device=x_reduced.device)
    receivers_nodes = row
    receivers_hedge = col

    receivers_pairs = torch.stack(
        (receivers_idx, receivers_nodes, receivers_hedge), dim=-1
    )
    key_func = lambda tup: tup[2]  # noqa: E731

    receivers_pairs = receivers_pairs.detach().cpu().numpy()
    receivers_pairs_sort = sorted(receivers_pairs, key=key_func)

    x_hedge = x_reduced.reshape(x_reduced.shape[0] * d, x_reduced.shape[-1])
    p = x_hedge @ rv
    p = p.squeeze(-1)
    p = p.reshape(row.shape[0], d)
    p = torch.transpose(p, 0, 1)
    p_1 = p.unsqueeze(-1)
    p_2 = p.unsqueeze(-2)

    edges = []
    edges_idx = []

    p_1_np = p_1.detach().cpu().numpy()
    p_2_np = p_2.detach().cpu().numpy()

    for _, group in itertools.groupby(receivers_pairs_sort, key_func):
        hyperedge = np.array(list(group)).astype(int)
        hyperedge_nodes = hyperedge[:, 1]
        hyperedge_pos = hyperedge[:, 0]

        p_1_partial = p_1_np[:, hyperedge_pos]
        p_2_partial = p_2_np[:, :, hyperedge_pos]

        p_dist_partial = p_1_partial - p_2_partial
        p_dist_partial = np.transpose(p_dist_partial, (1, 2, 0))
        p_dist_partial = np.linalg.norm(p_dist_partial, axis=-1, ord=2)

        s, i = np.unravel_index(
            np.argmax(p_dist_partial), p_dist_partial.shape
        )

        supremum, infimum = hyperedge_nodes[s], hyperedge_nodes[i]
        s_idx, i_idx = hyperedge_pos[s], hyperedge_pos[i]

        c = 2 * len(hyperedge_pos) - 3

        edges.extend([[supremum, infimum], [infimum, supremum]])
        edges_idx.extend([[s_idx, i_idx], [i_idx, s_idx]])

        norm = c if m else len(hyperedge_pos)
        weights[(supremum, infimum)] = weights.get(
            (supremum, infimum), 0
        ) + float(1 / norm)
        weights[(infimum, supremum)] = weights.get(
            (infimum, supremum), 0
        ) + float(1 / norm)

        if m:
            for mediator_idx, mediator_e in zip(
                hyperedge_pos, hyperedge_nodes, strict=False
            ):
                if mediator_e != supremum and mediator_e != infimum:
                    edges.extend(
                        [
                            [supremum, mediator_e],
                            [mediator_e, supremum],
                            [infimum, mediator_e],
                            [mediator_e, infimum],
                        ]
                    )
                    edges_idx.extend(
                        [
                            [s_idx, mediator_idx],
                            [mediator_idx, s_idx],
                            [i_idx, mediator_idx],
                            [mediator_idx, i_idx],
                        ]
                    )
                    weights = update(supremum, infimum, mediator_e, weights, c)

    # Add the self loops: for each node aggregate sum_e F_v<e^T F_v<e.
    receivers_pairs_diag = (
        torch.stack((receivers_idx, receivers_nodes, receivers_hedge), dim=-1)
        .detach()
        .cpu()
        .numpy()
    )
    key_func_diag = lambda tup: tup[1]  # noqa: E731
    receivers_pairs_sort_diag = sorted(receivers_pairs_diag, key=key_func_diag)

    edges_idx_diag = []
    all_contained_hyperedges = []

    idx = 0
    for _, group in itertools.groupby(
        receivers_pairs_sort_diag, key_func_diag
    ):
        contained_hyperedges = np.array(list(group)).astype(int)
        node_idx = contained_hyperedges[:, 1][0]
        contained_hyperedges = contained_hyperedges[:, 0]

        edges.extend([[node_idx, node_idx]])
        edges_idx_diag.extend(
            list(zip(contained_hyperedges, contained_hyperedges, strict=False))
        )
        all_contained_hyperedges.extend([idx] * len(contained_hyperedges))
        idx = idx + 1

    edges_idx = torch.tensor(np.array(edges_idx).transpose()).to(
        x_reduced.device
    )
    edges_idx_diag = torch.tensor(np.array(edges_idx_diag).transpose()).to(
        x_reduced.device
    )
    all_contained_hyperedges = torch.tensor(
        np.array(all_contained_hyperedges).astype(int)
    ).to(x_reduced.device)
    hgcn_edges = torch.tensor(np.array(edges).transpose()).to(x_reduced.device)

    return edges_idx, edges_idx_diag, all_contained_hyperedges, hgcn_edges


def _expand_indices_diag(hgcn_edges, d, device):
    """Expand node-level edges to block indices for a diagonal sheaf.

    Parameters
    ----------
    hgcn_edges : torch.Tensor
        Graph edges (node ids) of shape ``[2, K]``.
    d : int
        Stalk dimension.
    device : torch.device
        Device on which to build the index tensor.

    Returns
    -------
    torch.Tensor
        Block indices of shape ``[2, d * K]``.
    """
    d_range = torch.arange(d, device=device).view(1, -1, 1).repeat(2, 1, 1)
    hgcn_edges = hgcn_edges.unsqueeze(1)
    hgcn_edges = d * hgcn_edges + d_range
    return hgcn_edges.permute(0, 2, 1).reshape(2, -1)


def _expand_indices_full(hgcn_edges, d, device):
    """Expand node-level edges to block indices for a full ``d x d`` sheaf.

    Parameters
    ----------
    hgcn_edges : torch.Tensor
        Graph edges (node ids) of shape ``[2, K]``.
    d : int
        Stalk dimension.
    device : torch.device
        Device on which to build the index tensor.

    Returns
    -------
    torch.Tensor
        Block indices of shape ``[2, d * d * K]``.
    """
    d_range = torch.arange(d, device=device)
    d_range_edges = d_range.repeat(d).view(-1, 1)
    d_range_nodes = d_range.repeat_interleave(d).view(-1, 1)
    hgcn_edges = hgcn_edges.unsqueeze(1)

    index_0 = d * hgcn_edges[0] + d_range_nodes
    index_0 = index_0.permute((1, 0)).reshape(1, -1)
    index_1 = d * hgcn_edges[1] + d_range_edges
    index_1 = index_1.permute((1, 0)).reshape(1, -1)
    return torch.concat((index_0, index_1), 0)


def SheafLaplacianDiag(H, m, d, edge_index, sheaf):
    """Assemble the block sheaf Laplacian for a diagonal sheaf.

    Parameters
    ----------
    H : torch.Tensor
        Node features of shape ``[num_nodes * d, f]``.
    m : bool
        Whether to use mediators.
    d : int
        Stalk dimension.
    edge_index : torch.Tensor
        Incidence index of shape ``[2, nnz]``.
    sheaf : torch.Tensor
        Diagonal restriction maps of shape ``[nnz, d]``.

    Returns
    -------
    torch.Tensor
        Laplacian indices of shape ``[2, *]``.
    torch.Tensor
        Laplacian values of shape ``[*]``.
    """
    F = sheaf
    num_nodes = H.shape[0] // d
    hidden = H.shape[-1]

    h_selected = H.view((num_nodes, d, -1))
    h_selected = torch.index_select(h_selected, dim=0, index=edge_index[0])
    x_reduced = h_selected.permute(0, 2, 1)
    sheaf_blocks = torch.diag_embed(sheaf, dim1=-2, dim2=-1)
    sheaf_blocks = (
        sheaf_blocks.unsqueeze(1).repeat(1, hidden, 1, 1).reshape(-1, d, d)
    )

    x_reduced = x_reduced.reshape(-1, d).unsqueeze(-1)
    x_reduced = torch.bmm(sheaf_blocks, x_reduced)
    x_reduced = x_reduced.reshape(-1, hidden, d)
    x_reduced = x_reduced.permute(0, 2, 1)

    edges_idx, edges_idx_diag, all_contained_hyperedges, hgcn_edges = (
        reduce_graph(x_reduced, m, d, edge_index)
    )

    f_source = torch.index_select(F, dim=0, index=edges_idx[0])
    f_dest = torch.index_select(F, dim=0, index=edges_idx[1])
    attributes = -f_source * f_dest

    f_source_diag = torch.index_select(F, dim=0, index=edges_idx_diag[0])
    f_dest_diag = torch.index_select(F, dim=0, index=edges_idx_diag[1])
    attributes_diag = f_source_diag * f_dest_diag
    attributes_diag = scatter_add(
        attributes_diag, all_contained_hyperedges, dim=0
    )
    attributes = torch.concat([attributes, attributes_diag], axis=0)

    h_sheaf_index = _expand_indices_diag(hgcn_edges, d, H.device)
    h_sheaf_attributes = attributes.view(-1)
    return h_sheaf_index, h_sheaf_attributes


def SheafLaplacianGeneral(H, m, d, edge_index, sheaf):
    """Assemble the block sheaf Laplacian for a general sheaf.

    Parameters
    ----------
    H : torch.Tensor
        Node features of shape ``[num_nodes * d, f]``.
    m : bool
        Whether to use mediators.
    d : int
        Stalk dimension.
    edge_index : torch.Tensor
        Incidence index of shape ``[2, nnz]``.
    sheaf : torch.Tensor
        Restriction maps of shape ``[nnz, d * d]``.

    Returns
    -------
    torch.Tensor
        Laplacian indices of shape ``[2, *]``.
    torch.Tensor
        Laplacian values of shape ``[*]``.
    """
    F = sheaf
    num_nodes = H.shape[0] // d
    hidden = H.shape[-1]

    h_selected = H.view((num_nodes, d, -1))
    h_selected = torch.index_select(h_selected, dim=0, index=edge_index[0])
    x_reduced = h_selected.view((h_selected.shape[0], d, -1))
    x_reduced = x_reduced.permute(0, 2, 1)

    sheaf_blocks = sheaf.view(sheaf.shape[0], d, d)
    sheaf_blocks = (
        sheaf_blocks.unsqueeze(1).repeat(1, hidden, 1, 1).view(-1, d, d)
    )

    x_reduced = x_reduced.reshape(-1, d).unsqueeze(-1)
    x_reduced = torch.bmm(sheaf_blocks, x_reduced)
    x_reduced = x_reduced.reshape(-1, hidden, d)
    x_reduced = x_reduced.permute(0, 2, 1)

    edges_idx, edges_idx_diag, all_contained_hyperedges, hgcn_edges = (
        reduce_graph(x_reduced, m, d, edge_index)
    )

    F = F.view(F.shape[0], d, d)
    f_source = torch.index_select(F, dim=0, index=edges_idx[0])
    f_dest = torch.index_select(F, dim=0, index=edges_idx[1])
    attributes = -1 * torch.bmm(f_source.transpose(1, 2), f_dest)

    f_source_diag = torch.index_select(F, dim=0, index=edges_idx_diag[0])
    f_dest_diag = torch.index_select(F, dim=0, index=edges_idx_diag[1])
    attributes_diag = torch.bmm(f_source_diag.transpose(1, 2), f_dest_diag)
    attributes_diag = scatter_add(
        attributes_diag, all_contained_hyperedges, dim=0
    )
    attributes = torch.concat([attributes, attributes_diag], axis=0)

    h_sheaf_index = _expand_indices_full(hgcn_edges, d, H.device)
    h_sheaf_attributes = attributes.view(-1)
    return h_sheaf_index, h_sheaf_attributes


def SheafLaplacianOrtho(H, m, d, edge_index, sheaf):
    """Assemble the block sheaf Laplacian for an orthogonal sheaf.

    Because the restriction maps are orthogonal, ``F^T F = I`` and the
    diagonal blocks reduce to identity matrices scattered per node.

    Parameters
    ----------
    H : torch.Tensor
        Node features of shape ``[num_nodes * d, f]``.
    m : bool
        Whether to use mediators.
    d : int
        Stalk dimension.
    edge_index : torch.Tensor
        Incidence index of shape ``[2, nnz]``.
    sheaf : torch.Tensor
        Orthogonal restriction maps of shape ``[nnz, d, d]``.

    Returns
    -------
    torch.Tensor
        Laplacian indices of shape ``[2, *]``.
    torch.Tensor
        Laplacian values of shape ``[*]``.
    """
    F = sheaf
    num_nodes = H.shape[0] // d
    hidden = H.shape[-1]

    h_selected = H.view((num_nodes, d, -1))
    h_selected = torch.index_select(h_selected, dim=0, index=edge_index[0])
    x_reduced = h_selected.view((h_selected.shape[0], d, -1))
    x_reduced = x_reduced.permute(0, 2, 1)

    sheaf_blocks = sheaf.view(sheaf.shape[0], d, d)
    sheaf_blocks = (
        sheaf_blocks.unsqueeze(1).repeat(1, hidden, 1, 1).view(-1, d, d)
    )

    x_reduced = x_reduced.reshape(-1, d).unsqueeze(-1)
    x_reduced = torch.bmm(sheaf_blocks, x_reduced)
    x_reduced = x_reduced.reshape(-1, hidden, d)
    x_reduced = x_reduced.permute(0, 2, 1)

    edges_idx, edges_idx_diag, all_contained_hyperedges, hgcn_edges = (
        reduce_graph(x_reduced, m, d, edge_index)
    )

    F = F.view(F.shape[0], d, d)
    f_source = torch.index_select(F, dim=0, index=edges_idx[0])
    f_dest = torch.index_select(F, dim=0, index=edges_idx[1])
    attributes = -1 * torch.bmm(f_source.transpose(1, 2), f_dest)

    attributes_diag = (
        torch.eye(d)
        .unsqueeze(0)
        .repeat((edges_idx_diag.shape[1], 1, 1))
        .to(H.device)
    )
    attributes_diag = scatter_add(
        attributes_diag, all_contained_hyperedges, dim=0
    )
    attributes = torch.concat([attributes, attributes_diag], axis=0)

    h_sheaf_index = _expand_indices_full(hgcn_edges, d, H.device)
    h_sheaf_attributes = attributes.view(-1)
    return h_sheaf_index, h_sheaf_attributes
